#!/usr/bin/env python3
"""
Automatic mask generation for all images in a frame directory using SAM2
No prompts required - generates masks automatically for all detected objects
All parameters defined in code - no command line arguments needed
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ===== Configuration - Modify these parameters =====
CONFIG = {
    # Input/Output paths
    "frames_dir": "volleylab_data/test_for_auto_mask",  # 修改为你的帧文件夹路径
    "output_dir": "volleylab_data/test_for_auto_mask/result",   # 输出目录
    
    # SAM2 model paths
    "sam2_checkpoint": "checkpoints/sam2.1_hiera_large.pt",
    "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    
    # Mask generation parameters
    "points_per_side": 32,              # 采样网格密度 (32x32)
    "points_per_batch": 64,             # 批处理点数
    "pred_iou_thresh": 0.8,             # IoU阈值
    "stability_score_thresh": 0.9,      # 稳定性得分阈值
    "stability_score_offset": 0.7,      # 稳定性偏移
    "crop_n_layers": 1,                 # 裁剪层数
    "box_nms_thresh": 0.7,              # NMS阈值
    "crop_n_points_downscale_factor": 2, # 裁剪点下采样因子
    "min_mask_region_area": 100.0,      # 最小mask区域面积
    "use_m2m": True,                    # 使用mask-to-mask优化
    
    # Output settings
    "save_individual_masks": True,       # 保存单独的mask文件
    "save_overlay_images": True,         # 保存叠加图像
    "save_mask_data": True,              # 保存mask数据
    "show_progress": True,               # 显示进度条
    
    # Image processing
    "supported_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
}

class AutoMaskGenerator:
    def __init__(self):
        self.config = CONFIG
        
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )
        
        # Load SAM2 model
        print("Loading SAM2 model...")
        self.sam2 = build_sam2(
            self.config["model_config"],
            self.config["sam2_checkpoint"],
            device=self.device,
            apply_postprocessing=False
        )
        
        # Initialize mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=self.config["points_per_side"],
            points_per_batch=self.config["points_per_batch"],
            pred_iou_thresh=self.config["pred_iou_thresh"],
            stability_score_thresh=self.config["stability_score_thresh"],
            stability_score_offset=self.config["stability_score_offset"],
            crop_n_layers=self.config["crop_n_layers"],
            box_nms_thresh=self.config["box_nms_thresh"],
            crop_n_points_downscale_factor=self.config["crop_n_points_downscale_factor"],
            min_mask_region_area=self.config["min_mask_region_area"],
            use_m2m=self.config["use_m2m"],
        )
        print("✅ SAM2 model loaded successfully")
    
    def get_image_files(self, frames_dir):
        """Get all image files from the frames directory"""
        image_files = []
        frames_path = Path(frames_dir)
        
        if not frames_path.exists():
            raise ValueError(f"Frames directory does not exist: {frames_dir}")
        
        for file_path in frames_path.iterdir():
            if file_path.suffix.lower() in self.config["supported_formats"]:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def show_anns(self, anns, borders=True):
        """Display annotations overlay"""
        if len(anns) == 0:
            return np.zeros((100, 100, 4))
        
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                      sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        
        np.random.seed(42)  # 固定随机种子以获得一致的颜色
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        
        return img
    
    def save_individual_masks(self, masks, output_dir, image_name):
        """Save each mask as individual binary image"""
        masks_dir = output_dir / "individual_masks" / image_name.stem
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            mask_image = (mask * 255).astype(np.uint8)
            
            # Save mask
            mask_path = masks_dir / f"mask_{idx:03d}.png"
            Image.fromarray(mask_image).save(mask_path)
    
    def save_overlay_image(self, image, masks, output_dir, image_name):
        """Save image with masks overlay"""
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Create overlay
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        
        overlay = self.show_anns(masks)
        plt.imshow(overlay)
        plt.axis('off')
        
        # Save
        output_path = overlay_dir / f"{image_name.stem}_overlay.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0)
        plt.close()
    
    def save_mask_data(self, masks, output_dir, image_name):
        """Save mask data as numpy array"""
        data_dir = output_dir / "mask_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw mask data
        output_path = data_dir / f"{image_name.stem}_masks.npz"
        
        # Extract mask arrays and metadata
        mask_arrays = [mask['segmentation'] for mask in masks]
        areas = [mask['area'] for mask in masks]
        bboxes = [mask['bbox'] for mask in masks]
        ious = [mask['predicted_iou'] for mask in masks]
        stability_scores = [mask['stability_score'] for mask in masks]
        point_coords = [mask['point_coords'] for mask in masks]
        
        np.savez_compressed(
            output_path,
            masks=np.array(mask_arrays),
            areas=np.array(areas),
            bboxes=np.array(bboxes),
            predicted_ious=np.array(ious),
            stability_scores=np.array(stability_scores),
            point_coords=np.array(point_coords),
            num_masks=len(masks)
        )
    
    def process_single_image(self, image_path, output_dir):
        """Process a single image and generate masks"""
        try:
            # Load image
            image = Image.open(image_path)
            image_np = np.array(image.convert("RGB"))
            
            print(f"Processing: {image_path.name} ({image_np.shape})")
            
            # Generate masks
            masks = self.mask_generator.generate(image_np)
            
            # Save results
            image_name = Path(image_path)
            
            if self.config["save_individual_masks"]:
                self.save_individual_masks(masks, output_dir, image_name)
            
            if self.config["save_overlay_images"]:
                self.save_overlay_image(image_np, masks, output_dir, image_name)
            
            if self.config["save_mask_data"]:
                self.save_mask_data(masks, output_dir, image_name)
            
            print(f"  Generated {len(masks)} masks")
            return len(masks)
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return 0
    
    def process_frame_directory(self):
        """Process all images in the frames directory"""
        frames_dir = Path(self.config["frames_dir"])
        output_dir = Path(self.config["output_dir"])
        
        print(f"🔍 Looking for images in: {frames_dir}")
        
        # Get all image files
        image_files = self.get_image_files(frames_dir)
        
        if not image_files:
            print(f"❌ No image files found in {frames_dir}")
            print(f"Supported formats: {self.config['supported_formats']}")
            return
        
        print(f"📸 Found {len(image_files)} images to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        total_masks = 0
        successful_images = 0
        
        if self.config["show_progress"]:
            progress_bar = tqdm(image_files, desc="Processing images")
        else:
            progress_bar = image_files
        
        for image_path in progress_bar:
            num_masks = self.process_single_image(image_path, output_dir)
            if num_masks > 0:
                total_masks += num_masks
                successful_images += 1
            
            if self.config["show_progress"]:
                progress_bar.set_postfix({
                    "masks": num_masks, 
                    "total": total_masks,
                    "success": f"{successful_images}/{len(image_files)}"
                })
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"✅ Processing completed!")
        print(f"📊 Results Summary:")
        print(f"   • Total images found: {len(image_files)}")
        print(f"   • Successfully processed: {successful_images}")
        print(f"   • Total masks generated: {total_masks}")
        if successful_images > 0:
            print(f"   • Average masks per image: {total_masks / successful_images:.2f}")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        print("="*60)
        
        # Save summary file
        self.save_summary(len(image_files), successful_images, total_masks, output_dir)
    
    def save_summary(self, total_images, successful_images, total_masks, output_dir):
        """Save processing summary"""
        summary_path = output_dir / "processing_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== SAM2 Automatic Mask Generation Summary ===\n\n")
            f.write(f"Processing Date: {torch.utils.data.get_worker_info()}\n")
            f.write(f"Input Directory: {self.config['frames_dir']}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            f.write("Results:\n")
            f.write(f"  Total images found: {total_images}\n")
            f.write(f"  Successfully processed: {successful_images}\n")
            f.write(f"  Failed images: {total_images - successful_images}\n")
            f.write(f"  Total masks generated: {total_masks}\n")
            if successful_images > 0:
                f.write(f"  Average masks per image: {total_masks / successful_images:.2f}\n")
            
            f.write(f"\nConfiguration Used:\n")
            for section in ["Mask generation parameters", "Output settings"]:
                f.write(f"\n{section}:\n")
                
            # Mask generation parameters
            mask_params = [
                "points_per_side", "points_per_batch", "pred_iou_thresh", 
                "stability_score_thresh", "stability_score_offset", "crop_n_layers",
                "box_nms_thresh", "crop_n_points_downscale_factor", 
                "min_mask_region_area", "use_m2m"
            ]
            
            for param in mask_params:
                f.write(f"  {param}: {self.config[param]}\n")
            
            # Output settings
            output_params = [
                "save_individual_masks", "save_overlay_images", "save_mask_data"
            ]
            
            f.write(f"\nOutput settings:\n")
            for param in output_params:
                f.write(f"  {param}: {self.config[param]}\n")
            
            f.write(f"\nSupported image formats: {self.config['supported_formats']}\n")
        
        print(f"📄 Summary saved to: {summary_path}")


def main():
    print("🚀 Starting SAM2 Automatic Mask Generation")
    print("="*60)
    
    # Display configuration
    print("📋 Configuration:")
    print(f"   Input directory: {CONFIG['frames_dir']}")
    print(f"   Output directory: {CONFIG['output_dir']}")
    print(f"   Points per side: {CONFIG['points_per_side']}")
    print(f"   IoU threshold: {CONFIG['pred_iou_thresh']}")
    print(f"   Stability threshold: {CONFIG['stability_score_thresh']}")
    print(f"   Min region area: {CONFIG['min_mask_region_area']}")
    print("="*60)
    
    # Validate input directory
    if not Path(CONFIG['frames_dir']).exists():
        print(f"❌ Error: Input directory does not exist: {CONFIG['frames_dir']}")
        print("💡 Please modify the 'frames_dir' path in the CONFIG section of this script")
        return
    
    try:
        # Create mask generator and process
        generator = AutoMaskGenerator()
        generator.process_frame_directory()
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Please check your SAM2 model paths and input directory")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Automatic mask generation for all images in a frame directory using SAM2.
No prompts required - generates masks automatically for all detected objects.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

# If using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ===== Configuration - Modify these parameters =====
CONFIG = {
    # Input/Output paths
    "frames_dir": "volleylab_data/test_for_auto_mask",  # Path to your image frames directory
    "output_dir": "volleylab_data/test_for_auto_mask/result",   # Output directory for results

    # SAM2 model paths
    "sam2_checkpoint": "checkpoints/sam2.1_hiera_large.pt",
    "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",

    # Mask generation parameters
    "points_per_side": 32,              # Grid density for sampling points (e.g., 32x32)
    "points_per_batch": 64,             # Number of points processed in a batch
    "pred_iou_thresh": 0.8,             # IoU threshold for predicted masks
    "stability_score_thresh": 0.9,      # Stability score threshold for mask quality
    "stability_score_offset": 0.7,      # Stability score offset
    "crop_n_layers": 1,                 # Number of layers for cropping
    "box_nms_thresh": 0.7,              # Non-Maximum Suppression (NMS) threshold for bounding boxes
    "crop_n_points_downscale_factor": 2, # Downscale factor for crop points
    "min_mask_region_area": 100.0,      # Minimum area for a valid mask region
    "use_m2m": True,                    # Use mask-to-mask optimization

    # Output settings
    "save_individual_masks": True,      # Save each generated mask as a separate image
    "save_overlay_images": True,        # Save images with masks overlaid
    "save_mask_data": True,             # Save raw mask data (segmentation, bbox, etc.)
    "show_progress": True,              # Display a progress bar during processing

    # Image file formats
    "supported_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
}

class AutoMaskGenerator:
    def __init__(self):
        self.config = CONFIG

        # Set up device (CUDA/MPS/CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Configure CUDA for mixed precision if available
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8: # For newer GPUs (Ampere and later)
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

        # Initialize automatic mask generator with configured parameters
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
        """Get all image files from the specified frames directory."""
        image_files = []
        frames_path = Path(frames_dir)

        if not frames_path.exists():
            raise ValueError(f"Frames directory does not exist: {frames_dir}")

        for file_path in frames_path.iterdir():
            if file_path.suffix.lower() in self.config["supported_formats"]:
                image_files.append(file_path)

        return sorted(image_files)

    def show_anns(self, anns, borders=True):
        """Display annotations as an overlay on an image."""
        if len(anns) == 0:
            return np.zeros((100, 100, 4)) # Return a transparent image if no annotations

        # Sort annotations by area (largest first)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        # Create a blank image for the overlay
        img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                       sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0 # Set alpha channel to 0 for full transparency initially

        np.random.seed(42)  # Fix random seed for consistent mask colors
        for ann in sorted_anns:
            m = ann['segmentation'] # Binary mask for the current annotation
            # Generate a random color with 50% transparency
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask # Apply color to the mask area

            if borders:
                # Find contours for mask borders and draw them
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) # Blue border

        return img

    def save_individual_masks(self, masks, output_dir, image_name):
        """Save each generated mask as a separate binary PNG image."""
        masks_dir = output_dir / "individual_masks" / image_name.stem
        masks_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

        for idx, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            mask_image = (mask * 255).astype(np.uint8) # Convert boolean mask to 0-255 grayscale

            # Save the mask image
            mask_path = masks_dir / f"mask_{idx:03d}.png"
            Image.fromarray(mask_image).save(mask_path)

    def save_overlay_image(self, image, masks, output_dir, image_name):
        """Save the original image with generated masks overlaid."""
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

        # Create a matplotlib figure to draw the image and overlay
        plt.figure(figsize=(12, 12))
        plt.imshow(image)

        # Generate the overlay image using show_anns method
        overlay = self.show_anns(masks)
        plt.imshow(overlay) # Display the overlay
        plt.axis('off') # Turn off axis labels

        # Save the figure
        output_path = overlay_dir / f"{image_name.stem}_overlay.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0)
        plt.close() # Close the plot to free memory

    def save_mask_data(self, masks, output_dir, image_name):
        """Save raw mask data (segmentation, bbox, etc.) for each image as a compressed NPZ file."""
        data_dir = output_dir / "mask_data"
        data_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

        output_path = data_dir / f"{image_name.stem}_masks.npz"

        # Extract relevant data from each mask dictionary
        mask_arrays = [mask['segmentation'] for mask in masks]
        areas = [mask['area'] for mask in masks]
        bboxes = [mask['bbox'] for mask in masks]
        ious = [mask['predicted_iou'] for mask in masks]
        stability_scores = [mask['stability_score'] for mask in masks]
        point_coords = [mask['point_coords'] for mask in masks]

        # Save all extracted data into a single compressed NPZ file
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
        """Process a single image: load, generate masks, and save results."""
        try:
            # Load image and convert to RGB
            image = Image.open(image_path)
            image_np = np.array(image.convert("RGB"))

            print(f"Processing: {image_path.name} ({image_np.shape})")

            # Generate masks automatically
            masks = self.mask_generator.generate(image_np)

            # Save results based on configuration
            image_name = Path(image_path)

            if self.config["save_individual_masks"]:
                self.save_individual_masks(masks, output_dir, image_name)

            if self.config["save_overlay_images"]:
                self.save_overlay_image(image_np, masks, output_dir, image_name)

            if self.config["save_mask_data"]:
                self.save_mask_data(masks, output_dir, image_name)

            print(f"  Generated {len(masks)} masks")
            return len(masks) # Return number of masks generated
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return 0 # Return 0 masks on error

    def process_frame_directory(self):
        """Process all images in the configured frames directory."""
        frames_dir = Path(self.config["frames_dir"])
        output_dir = Path(self.config["output_dir"])

        print(f"🔍 Looking for images in: {frames_dir}")

        # Get list of image files
        image_files = self.get_image_files(frames_dir)

        if not image_files:
            print(f"❌ No image files found in {frames_dir}")
            print(f"Supported formats: {self.config['supported_formats']}")
            return

        print(f"📸 Found {len(image_files)} images to process")

        # Create main output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        total_masks = 0
        successful_images = 0

        # Use tqdm for progress bar if enabled
        if self.config["show_progress"]:
            progress_bar = tqdm(image_files, desc="Processing images")
        else:
            progress_bar = image_files

        for image_path in progress_bar:
            num_masks = self.process_single_image(image_path, output_dir)
            if num_masks > 0:
                total_masks += num_masks
                successful_images += 1

            # Update progress bar description
            if self.config["show_progress"]:
                progress_bar.set_postfix({
                    "masks": num_masks,
                    "total": total_masks,
                    "success": f"{successful_images}/{len(image_files)}"
                })

        # Print final summary
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
        """Save a summary of the processing results to a text file."""
        summary_path = output_dir / "processing_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== SAM2 Automatic Mask Generation Summary ===\n\n")
            f.write(f"Processing Date: {torch.utils.data.get_worker_info()}\n") # Note: get_worker_info() might not be ideal here for single thread
            f.write(f"Input Directory: {self.config['frames_dir']}\n")
            f.write(f"Output Directory: {output_dir}\n\n")

            f.write("Results:\n")
            f.write(f"   Total images found: {total_images}\n")
            f.write(f"   Successfully processed: {successful_images}\n")
            f.write(f"   Failed images: {total_images - successful_images}\n")
            f.write(f"   Total masks generated: {total_masks}\n")
            if successful_images > 0:
                f.write(f"   Average masks per image: {total_masks / successful_images:.2f}\n")

            f.write(f"\nConfiguration Used:\n")
            # Write mask generation parameters
            f.write(f"Mask generation parameters:\n")
            mask_params = [
                "points_per_side", "points_per_batch", "pred_iou_thresh",
                "stability_score_thresh", "stability_score_offset", "crop_n_layers",
                "box_nms_thresh", "crop_n_points_downscale_factor",
                "min_mask_region_area", "use_m2m"
            ]
            for param in mask_params:
                f.write(f"   {param}: {self.config[param]}\n")

            # Write output settings
            f.write(f"\nOutput settings:\n")
            output_params = [
                "save_individual_masks", "save_overlay_images", "save_mask_data"
            ]
            for param in output_params:
                f.write(f"   {param}: {self.config[param]}\n")

            f.write(f"\nSupported image formats: {self.config['supported_formats']}\n")

        print(f"📄 Summary saved to: {summary_path}")


def main():
    print("🚀 Starting SAM2 Automatic Mask Generation")
    print("="*60)

    # Display configuration for user review
    print("📋 Configuration:")
    print(f"   Input directory: {CONFIG['frames_dir']}")
    print(f"   Output directory: {CONFIG['output_dir']}")
    print(f"   Points per side: {CONFIG['points_per_side']}")
    print(f"   IoU threshold: {CONFIG['pred_iou_thresh']}")
    print(f"   Stability threshold: {CONFIG['stability_score_thresh']}")
    print(f"   Min region area: {CONFIG['min_mask_region_area']}")
    print("="*60)

    # Validate input directory existence
    if not Path(CONFIG['frames_dir']).exists():
        print(f"❌ Error: Input directory does not exist: {CONFIG['frames_dir']}")
        print("💡 Please modify the 'frames_dir' path in the CONFIG section of this script")
        return

    try:
        # Create mask generator instance and start processing
        generator = AutoMaskGenerator()
        generator.process_frame_directory()

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Please check your SAM2 model paths and input directory")


if __name__ == "__main__":
    main()
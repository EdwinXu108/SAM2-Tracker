#!/usr/bin/env python3
"""
SAM2 Video Tracking Script
Supports points and box annotations
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor

# ===== Configuration =====
CONFIG = {
    "input_path": "volleylab_data/raw/output_BV1Ato6YKEWy_segment_4.mp4",
    "output_dir": "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_4",

    "checkpoint_path": "checkpoints/sam2.1_hiera_large.pt",
    "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",

    # Maximum number of frames to process. Set to None for all frames, or an integer for a subset
    "max_frames": None,
    # Whether to display result plots during execution (initial annotation results and sample propagation results)
    "show_results": True,
}

# ===== Annotation Definitions =====
ANNOTATIONS = [
    # Point annotation format: (frame_idx, obj_id, "points", [[x1, y1], [x2, y2], ...], [label1, label2, ...])
    # Box annotation format: (frame_idx, obj_id, "box", [x1, y1, x2, y2], None)
    # Example: (100, 1, "box", [550, 110, 580, 140], None),  # x1,y1 is top-left, x2,y2 is bottom-right
    (0, 14, "box", [491, 241, 537, 357], None),
    (0, 15, "box", [533, 256, 591, 385], None),
    (0, 5, "box", [592, 323, 707, 483], None),
    (0, 7, "box", [674, 326, 785, 472], None),
    (9, 14, "box", [698, 267, 754, 379], None),
    (17, 17, "box", [801, 288, 899, 435], None),
    (21, 14, "box", [745, 277, 802, 385], None),
]

class SAM2Tracker:
    def __init__(self):
        # Automatically detect and set device to CUDA (GPU) or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Clear CUDA cache if using GPU to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading SAM2 model...")
        self.predictor = build_sam2_video_predictor(
            CONFIG["model_config"],
            CONFIG["checkpoint_path"],
            device=self.device
        )
        # Removed .float() call, relying on torch.amp.autocast for mixed precision
        print("✅ Model loaded successfully")

        self.inference_state = None
        self.frames_dir = None
        self.frame_names = []
        # self.video_segments will now only store sample results for show_results, not all frames
        self.video_segments = {} 

    def extract_frames(self, video_path, output_dir, max_frames=None):
        """
        Extracts frames from a video and saves them to the specified directory.
        Frame files will be named numerically (e.g., 00000.jpg, 00001.jpg).
        """
        print(f"Extracting frames from: {video_path}")
        os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        if max_frames is not None and total_frames > max_frames:
            # Calculate step to uniformly sample max_frames from the video
            step = total_frames // max_frames
            if step == 0: # Avoid division by zero if total_frames < max_frames, save all frames then
                step = 1
            processed_frames_limit = max_frames # Actual limit for frames to process
        else:
            step = 1
            processed_frames_limit = total_frames # If max_frames is None or insufficient, process all frames

        frame_idx = 0 # Original video frame index
        saved_count = 0 # Number of frames saved so far

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Video reading finished

            # Save frame if it's within the sampling step and we haven't reached max_frames limit
            if frame_idx % step == 0 and saved_count < processed_frames_limit:
                # Critical: Frame files named numerically, e.g., "00000.jpg", for SAM2 compatibility
                cv2.imwrite(f"{output_dir}/{saved_count:05d}.jpg", frame)
                saved_count += 1

            frame_idx += 1
            # Stop loop immediately if saved_count reaches the limit
            if saved_count >= processed_frames_limit:
                break

        cap.release()
        print(f"✅ Frame extraction completed: Saved {saved_count} frames")
        return saved_count

    def is_video_file(self, path):
        """Checks if the given path is a supported video file."""
        if not os.path.exists(path):
            return False
        if os.path.isdir(path):
            return False

        # List of supported video file extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        return Path(path).suffix.lower() in video_extensions

    def is_frames_directory(self, path):
        """
        Checks if the given path is a directory containing frame images
        that conform to SAM2's expected numeric naming (e.g., 00000.jpg).
        """
        if not os.path.isdir(path):
            return False

        # List of supported image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = os.listdir(path)

        # Check if at least one numerically named image file exists in the directory
        numeric_image_files = [
            f for f in files
            if Path(f).suffix.lower() in image_extensions and Path(f).stem.isdigit()
        ]

        return len(numeric_image_files) > 0

    def prepare_frames(self, input_path):
        """
        Prepares frames from the input path (either a video file or a frames directory).
        If the input is a video file, frames are extracted into a new directory.
        """
        if self.is_video_file(input_path):
            print(f"📹 Input is a video file: {input_path}")
            video_name = Path(input_path).stem # Get video filename without extension
            # Define the output directory for frames, placed next to the main output_dir
            frames_dir = Path(CONFIG["output_dir"]).parent / f"{video_name}_frames"

            os.makedirs(frames_dir, exist_ok=True) # Ensure frame output directory exists

            self.extract_frames(input_path, str(frames_dir), CONFIG["max_frames"])
            return str(frames_dir) # Return the string path to the frames directory

        elif self.is_frames_directory(input_path):
            print(f"📁 Input is a frames directory: {input_path}")
            return input_path # If it's already a frames directory, directly return the path

        else:
            raise ValueError(f"Input path is neither a valid video file nor a frames directory: {input_path}")

    def load_frames(self, frames_dir):
        """Loads frames and initializes SAM2's inference state."""
        self.frames_dir = frames_dir

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_files = os.listdir(frames_dir)

        # Filter image files and sort them numerically by filename (e.g., 00000.jpg, 00001.jpg)
        self.frame_names = sorted([
            f for f in all_files
            if Path(f).suffix.lower() in image_extensions and Path(f).stem.isdigit()
        ], key=lambda p: int(Path(p).stem)) # Sort by the integer value of the filename stem

        if not self.frame_names:
            raise ValueError(f"No numerically named image files found in directory: {frames_dir}. Expected format like '00000.jpg'.")

        print(f"Found {len(self.frame_names)} frame files")

        print("Initializing inference state...")
        # Use torch.amp.autocast to automatically handle mixed precision
        with torch.amp.autocast('cuda'): # Follow the new usage, explicitly specifying device as 'cuda'
            self.inference_state = self.predictor.init_state(
                video_path=frames_dir, # Pass the path to the frames directory
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )
        print("✅ Initialization completed")

    def add_annotation(self, frame_idx, obj_id, annotation_type, data, labels):
        """
        Adds an annotation to a specific frame and object ID. Supports point and box annotations.
        """
        print(f"Adding annotation: Frame {frame_idx}, Object {obj_id}, Type {annotation_type}")

        points = None
        box = None
        if annotation_type == "points":
            points = np.array(data, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
        elif annotation_type == "box":
            box = np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported annotation type: {annotation_type}")

        # Use torch.amp.autocast to automatically handle mixed precision
        with torch.amp.autocast('cuda'): # Follow the new usage, explicitly specifying device as 'cuda'
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box, # Pass box parameter, even if None; SAM2 will handle based on data
            )

        print(f"✅ Annotation added successfully: {out_obj_ids}")

        if CONFIG["show_results"]:
            # Display annotation results (if any), passing the first mask logits for display
            self.show_annotation_result(frame_idx, obj_id, annotation_type, data, labels, out_mask_logits[0])

        return out_obj_ids, out_mask_logits

    def show_annotation_result(self, frame_idx, obj_id, annotation_type, data, labels, mask_logits):
        """
        Displays annotation results on a single frame (original image, annotation, and predicted mask).
        """
        frame_path = os.path.join(self.frames_dir, self.frame_names[frame_idx])
        # Use PIL.Image to load the image and convert to RGB format for consistent colors
        image = np.array(Image.open(frame_path).convert('RGB'))

        plt.figure(figsize=(10, 6))
        plt.imshow(image)

        if annotation_type == "points":
            points = np.array(data)
            labels = np.array(labels) if labels is not None else np.ones(len(points))

            pos_points = points[labels == 1] # Positive sample points (foreground)
            neg_points = points[labels == 0] # Negative sample points (background)

            if len(pos_points) > 0:
                plt.scatter(pos_points[:, 0], pos_points[:, 1], c='green', marker='*', s=200, label='Positive')
            if len(neg_points) > 0:
                plt.scatter(neg_points[:, 0], neg_points[:, 1], c='red', marker='*', s=200, label='Negative')

        elif annotation_type == "box":
            # data is [x1, y1, x2, y2]
            x1, y1, x2, y2 = data
            width = x2 - x1
            height = y2 - y1
            # Draw a green bounding box
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='green', linewidth=2, label='Bounding Box')
            plt.gca().add_patch(rect)

        # Display the predicted mask
        mask = (mask_logits > 0.0).cpu().numpy() # Convert logits to a boolean mask
        h, w = mask.shape[-2:]
        # Create a semi-transparent green mask overlay (RGBA)
        mask_image = mask.reshape(h, w, 1) * np.array([0, 1, 0, 0.6]).reshape(1, 1, -1)
        plt.imshow(mask_image)

        plt.title(f"Frame {frame_idx} - Object {obj_id} ({annotation_type})")
        plt.legend()
        plt.axis('off') # Turn off axes
        plt.show()

    def propagate(self):
        """
        Propagates annotations and saves results directly to disk for each frame.
        This method replaces the need for a separate save_results call for all frames.
        """
        print("Starting propagation and saving results...")

        output_dir = Path(CONFIG["output_dir"])
        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"
        
        # Ensure output directories exist
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        processed_frame_count = 0

        # Use torch.amp.autocast to automatically handle mixed precision
        with torch.amp.autocast('cuda'): # Follow the new usage, explicitly specifying device as 'cuda'
            for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(self.predictor.propagate_in_video(self.inference_state)):
                # Skip if frame_idx is out of bounds (should not happen with proper frame_names list)
                if out_frame_idx >= len(self.frame_names):
                    print(f"Warning: Frame index {out_frame_idx} out of bounds for frame_names. Skipping save.")
                    continue

                frame_path = os.path.join(self.frames_dir, self.frame_names[out_frame_idx])
                # Load original image for overlay
                current_frame_image = np.array(Image.open(frame_path).convert('RGB'))

                # Store current frame's results temporarily for show_results if needed
                # For memory efficiency, we will only store results for frames chosen by show_results
                # during the actual propagation loop.
                if CONFIG["show_results"] and out_frame_idx % 50 == 0: # Store every 50th frame for sample display
                     self.video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                        for j, out_obj_id in enumerate(out_obj_ids)
                    }

                for j, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy()

                    # Save binary mask as PNG (0 or 255)
                    mask_binary = (mask.squeeze() * 255).astype(np.uint8)
                    Image.fromarray(mask_binary).save(masks_dir / f"frame_{out_frame_idx:05d}_obj_{out_obj_id}.png")

                    # Save overlay image
                    overlay = current_frame_image.copy()
                    mask_area = mask.squeeze().astype(bool) # Get boolean region of the mask
                    # Overlay green color onto the mask area, preserving some original texture
                    overlay[mask_area] = overlay[mask_area] * 0.7 + np.array([0, 255, 0]) * 0.3
                    Image.fromarray(overlay.astype(np.uint8)).save(overlays_dir / f"frame_{out_frame_idx:05d}_obj_{out_obj_id}.jpg")
                
                processed_frame_count += 1

                # Clear CUDA cache periodically to prevent memory accumulation
                if (i + 1) % 20 == 0 and torch.cuda.is_available(): # Clear every 20 frames
                    torch.cuda.empty_cache()
                    print(f"Cleared CUDA cache, processed up to frame {out_frame_idx}")
                
        print(f"✅ Propagation and saving completed: Processed {processed_frame_count} frames")

    def show_results(self, frame_indices=None):
        """
        Displays tracking results for multiple frames, including original image and predicted masks.
        By default, displays the start, 1/4, 1/2, 3/4, and end frames.
        Note: This method will now load results from disk for display, if not already in self.video_segments.
        """
        if frame_indices is None:
            total_frames = len(self.frame_names)
            if total_frames > 0:
                # Select representative frames for display
                frame_indices = sorted(list(set([0,
                                                total_frames // 4,
                                                total_frames // 2,
                                                total_frames * 3 // 4,
                                                total_frames - 1])))
                # Filter out invalid or duplicate indices for very short videos
                frame_indices = [idx for idx in frame_indices if idx >= 0 and idx < total_frames]
            else:
                print("No frames to show results for.")
                return

        # Prepare directories for loading results
        output_dir = Path(CONFIG["output_dir"])
        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"

        plt.figure(figsize=(15, 6)) # Set overall figure size
        # Iterate through selected frame indices, displaying up to 6 subplots (2 rows x 3 columns)
        for i, frame_idx in enumerate(frame_indices[:6]):
            if i >= 6: break # Ensure no more than 6 subplots

            plt.subplot(2, 3, i + 1) # Set subplot position

            frame_path = os.path.join(self.frames_dir, self.frame_names[frame_idx])
            image = np.array(Image.open(frame_path).convert('RGB')) # Load and convert to RGB
            plt.imshow(image) # Display original image
            
            # Find any overlay for this frame if multiple objects
            overlay_files = list(overlays_dir.glob(f"frame_{frame_idx:05d}_obj_*.jpg"))
            if overlay_files:
                
                # If we've stored segments in self.video_segments during propagate (for sample frames)
                if frame_idx in self.video_segments:
                    for obj_id, mask in self.video_segments[frame_idx].items():
                        h, w = mask.shape[-2:]
                        mask_image = mask.reshape(h, w, 1) * np.array([0, 1, 0, 0.6]).reshape(1, 1, -1)
                        plt.imshow(mask_image)
                    plt.title(f"Frame {frame_idx} ✓") # Title indicates results are present
                else:
                    # Fallback: if not in video_segments, try loading a generic overlay or indicate missing
                    # This path indicates that the frame's results were saved but not kept in memory for display
                    plt.title(f"Frame {frame_idx} (saved to disk)") 
            else:
                plt.title(f"Frame {frame_idx} ✗") # Title indicates no results

            plt.axis('off') # Turn off axes

        plt.tight_layout() # Adjust subplot layout automatically
        plt.show() # Display the plot


def main():
    print("SAM2 Video Tracking Script")

    tracker = SAM2Tracker()
    
    # Automatically handle input path (video file or frames directory)
    frames_dir = tracker.prepare_frames(CONFIG["input_path"])
    tracker.load_frames(frames_dir)

    # Add initial annotations
    print("Adding annotations...")
    for annotation in ANNOTATIONS:
        frame_idx, obj_id, annotation_type, data, labels = annotation
        tracker.add_annotation(frame_idx, obj_id, annotation_type, data, labels)

    # Propagate annotations and save results directly
    tracker.propagate()

    if CONFIG["show_results"]:
        tracker.show_results()

    print("🎉 Completed!")

if __name__ == "__main__":
    main()
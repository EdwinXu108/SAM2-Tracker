import cv2
import numpy as np
import os

def create_masked_video(mask_dir, original_frame_dir, output_video_path, obj_id, mask_mode='original_color', fps=30):
    """
    根据mask和原始帧合成视频，并对指定obj_id的区域进行特殊处理。

    Args:
        mask_dir (str): 包含mask图像的目录路径 (e.g., 'path/to/masks').
        original_frame_dir (str): 包含原始帧图像的目录路径 (e.g., 'path/to/frames').
        output_video_path (str): 输出视频的完整文件路径及文件名 (e.g., 'output/my_video.mp4').
        obj_id (int): 指定需要处理的obj ID。
        mask_mode (str): mask区域内的显示模式。
                         'original_color' (默认): mask区域显示原始颜色。
                         'white': mask区域显示为全白。
        fps (int): 输出视频的帧率。
    """
    print(f"Starting video creation for obj_id: {obj_id} with mask_mode: {mask_mode}")
    print(f"Masks from: {mask_dir}")
    print(f"Original frames from: {original_frame_dir}")
    print(f"Output video to: {output_video_path}")

    # 获取所有原始帧文件，并按数字顺序排序
    # 注意：你的原始帧是 '00000.jpg', '00001.jpg' 等，这里需要调整提取数字的逻辑
    original_frames = sorted([f for f in os.listdir(original_frame_dir) if f.endswith('.jpg')],
                             key=lambda x: int(os.path.splitext(x)[0])) # 提取不带扩展名的数字部分进行排序

    if not original_frames:
        print(f"Error: No original frame (JPG) images found in {original_frame_dir}")
        return

    # 获取第一帧图像以确定视频尺寸
    first_frame_path = os.path.join(original_frame_dir, original_frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Could not read the first frame: {first_frame_path}. Please check file existence and format.")
        return

    height, width, _ = first_frame.shape
    print(f"Video dimensions: {width}x{height}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 定义视频编码器和VideoWriter对象
    # 根据你的系统和OpenCV安装，你可以尝试不同的编码器，例如 'mp4v' 或 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}. Check file path, codec, or permissions.")
        # 尝试打印支持的编码器 (这可能在某些OpenCV版本上不可用)
        # print(cv2.getBuildInformation())
        return

    frame_count = 0
    for original_frame_name in original_frames:
        # 原始帧的文件名是 '00000.jpg'，我们需要 '00000' 来构建mask文件名
        frame_number_padded = os.path.splitext(original_frame_name)[0] # 提取 '00000'

        original_path = os.path.join(original_frame_dir, original_frame_name)
        
        # 构建对应的mask文件名: 'frame_00000_obj_8.png'
        mask_filename = f"frame_{frame_number_padded}_obj_{obj_id}.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        original_image = cv2.imread(original_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 读取mask为灰度图

        if original_image is None:
            print(f"Warning: Could not read original frame: {original_path}, skipping.")
            continue
        # 如果mask不存在，则将该帧处理为全黑
        if mask_image is None:
            print(f"Warning: Could not find or read mask for obj_id {obj_id} at {mask_path}. Frame will be black.")
            processed_frame = np.zeros_like(original_image)
            out.write(processed_frame)
            frame_count += 1
            continue

        # 确保mask和原始图像尺寸相同
        if original_image.shape[:2] != mask_image.shape[:2]:
            print(f"Warning: Size mismatch between original frame ({original_image.shape[:2]}) and mask ({mask_image.shape[:2]}) for {original_frame_name}. Resizing mask.")
            mask_image = cv2.resize(mask_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # 创建一个全黑的背景
        black_background = np.zeros_like(original_image)

        # 将mask二值化，确保只有0和255
        _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY) 

        # 根据mask_mode处理前景区域
        if mask_mode == 'white':
            foreground = np.full_like(original_image, 255) # 全白图像
            processed_frame = cv2.bitwise_and(foreground, foreground, mask=binary_mask)
        elif mask_mode == 'original_color':
            processed_frame = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
        else:
            print(f"Error: Invalid mask_mode '{mask_mode}'. Using 'original_color' as default.")
            processed_frame = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

        # 将处理后的前景区域叠加到黑色背景上
        final_frame = cv2.add(black_background, processed_frame)
        
        out.write(final_frame)
        frame_count += 1

    out.release()
    print(f"Video creation complete. Total frames processed: {frame_count}. Video saved as {output_video_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义你的输入和输出路径
    # 请根据你的实际文件路径进行修改
    
    # 你的mask图片所在的文件夹路径
    input_mask_directory = "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1/masks"
    
    # 你的原始帧图片所在的文件夹路径 (包含00000.jpg, 00001.jpg 等)
    input_original_frames_directory = "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1_frames"
    
    # 你希望输出视频的完整路径和文件名
    # 建议创建一个'output'文件夹来存放结果，例如:
    output_video_file_path = "output_videos/processed_obj_17_video_white.mp4" 

    desired_obj_id = 17 # 你想处理的 obj ID
    
    # 模式选择：'original_color'（保留原始颜色） 或 'white'（变为纯白色）
    processing_mode = 'white' 
    
    # 视频帧率
    video_fps = 25 

    create_masked_video(
        input_mask_directory,
        input_original_frames_directory,
        output_video_file_path,
        desired_obj_id,
        mask_mode=processing_mode,
        fps=video_fps
    )
import cv2
import os
import numpy as np
from pathlib import Path
from PIL import Image # 用于更健壮的图像加载

def create_colored_mask_overlay_video(
    original_frames_dir,
    masks_base_dir,
    output_video_path,
    fps=30,
    alpha=0.5 # 叠加透明度 (0.0 到 1.0)
):
    """
    将目录中不同物体的独立掩码合并到一起，并叠加到原始视频帧上，
    为每个物体赋予独特的颜色，最终生成一个 MP4 视频。

    Args:
        original_frames_dir (str): 包含原始视频帧的目录路径
                                   (例如: '00000.jpg', '00001.jpg')。
        masks_base_dir (str): 包含 'masks' 子文件夹的根目录路径
                              (例如: 'volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1'，
                               此目录下应有 'masks' 文件夹)。
        output_video_path (str): 输出 MP4 视频文件的完整路径。
        fps (int): 输出视频的帧率。
        alpha (float): 彩色掩码叠加的透明度 (0.0 表示完全透明，1.0 表示完全不透明)。
    """

    # 构造掩码文件夹的完整路径
    masks_dir = Path(masks_base_dir) / "masks"

    # 路径有效性检查
    if not original_frames_dir:
        raise ValueError("原始帧目录路径不能为空。")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"掩码目录未找到: {masks_dir}")
    if not Path(original_frames_dir).is_dir():
        raise FileNotFoundError(f"原始帧目录未找到: {original_frames_dir}")

    # 1. 获取原始帧文件列表并按数字顺序排序
    # 假设原始帧命名如 "00000.jpg"
    original_frame_files = sorted([f for f in os.listdir(original_frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                                   key=lambda x: int(Path(x).stem))

    if not original_frame_files:
        print(f"在 {original_frames_dir} 中未找到原始帧文件。退出。")
        return

    # 读取第一帧以获取视频的尺寸 (高、宽)
    first_frame_path = os.path.join(original_frames_dir, original_frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise IOError(f"无法读取第一帧: {first_frame_path}。请检查文件完整性。")

    height, width, _ = first_frame.shape

    # 定义视频编码器并创建 VideoWriter 对象
    # 'mp4v' 是适用于 .mp4 文件的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise IOError(f"无法打开视频写入器以创建 {output_video_path}。请检查路径/权限/编码器。")

    print(f"正在创建视频: {output_video_path}，共 {len(original_frame_files)} 帧，帧率 {fps} FPS。")
    print(f"视频分辨率: {width}x{height}")

    # 用于存储每个物体 ID 对应颜色的字典，确保颜色一致性
    # 颜色将以 BGR 格式存储，以适应 OpenCV
    object_colors = {}
    
    # 预定义一组颜色 (BGR 格式)，用于循环分配给不同物体，以便更好区分
    color_palette = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 品红色
        (255, 255, 0),  # 青色
        (0, 165, 255),  # 橙色
        (128, 0, 128),  # 紫色
        (0, 128, 0),    # 暗绿色
        (128, 128, 0),  # 橄榄色
        (0, 0, 128),    # 暗蓝色
        (128, 0, 0),    # 栗色
    ]
    color_idx = 0 # 颜色索引，用于从色板中选取颜色

    # 遍历所有原始视频帧
    for i, frame_filename in enumerate(original_frame_files):
        # 从文件名中提取帧索引 (例如: "00000.jpg" -> 0)
        frame_idx = int(Path(frame_filename).stem)

        original_frame_path = os.path.join(original_frames_dir, frame_filename)
        # 使用 PIL 读取图像，然后转换为 OpenCV 的 BGR 格式
        base_image = np.array(Image.open(original_frame_path).convert('RGB'))
        base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR) # 将 PIL 的 RGB 转换为 OpenCV 的 BGR

        # 创建一个与原始帧大小相同的空白叠加层，用于叠加当前帧的所有掩码
        overlay_frame = np.zeros_like(base_image, dtype=np.uint8)

        # 查找当前帧对应的所有掩码文件
        # 文件名格式例如: "frame_00000_obj_7.png"
        mask_files = sorted(masks_dir.glob(f"frame_{frame_idx:05d}_obj_*.png"))

        # 遍历当前帧的所有物体掩码文件
        for mask_file in mask_files:
            # 从掩码文件名中提取物体 ID (例如: "frame_00000_obj_7.png" -> "7")
            try:
                obj_id = int(str(mask_file.stem).split('_obj_')[-1])
            except ValueError:
                print(f"警告: 无法从文件名 {mask_file.name} 中解析物体 ID。跳过此掩码。")
                continue

            # 如果该物体 ID 尚未分配颜色，则分配一个新颜色
            if obj_id not in object_colors:
                object_colors[obj_id] = color_palette[color_idx % len(color_palette)]
                color_idx += 1
            
            # 读取掩码文件为灰度图 (0 或 255)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码文件 {mask_file}。跳过此掩码。")
                continue

            # 确保掩码是二值图像 (值只有 0 或 255)
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)

            # 根据物体 ID 的颜色，为当前掩码着色
            colored_mask = np.zeros_like(base_image, dtype=np.uint8)
            color = object_colors[obj_id]
            # 仅在掩码值为 255 的区域应用颜色
            colored_mask[mask == 255] = color

            # 将这个着色后的掩码添加到当前帧的总叠加层上
            # cv2.addWeighted 用于图像混合，这里是简单相加
            overlay_frame = cv2.addWeighted(overlay_frame, 1, colored_mask, 1, 0)
        
        # 最后，将合并了所有物体掩码的叠加层，与原始图像进行融合（按透明度 alpha）
        # (1 - alpha) 用于原始图像的权重，alpha 用于叠加层的权重
        final_frame = cv2.addWeighted(base_image, 1 - alpha, overlay_frame, alpha, 0)

        out.write(final_frame) # 将处理后的帧写入视频文件

        # 打印进度
        if (i + 1) % 50 == 0 or (i + 1) == len(original_frame_files):
            print(f"已处理 {i+1}/{len(original_frame_files)} 帧，用于视频合成。")

    out.release() # 释放 VideoWriter 对象
    print(f"✅ 视频创建成功: {output_video_path}")

# --- 如何使用 ---
if __name__ == "__main__":
    # --- 视频合成配置 ---
    # 请替换为你的原始视频帧所在的目录路径 (由 SAM2 Tracker 提取而来)
    ORIGINAL_FRAMES_DIRECTORY = "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1_frames"
    
    # 请替换为 SAM2 Tracker 保存掩码的根输出目录
    # 此目录应该包含名为 'masks' 的子文件夹
    MASKS_OUTPUT_BASE_DIRECTORY = "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1" 
    
    # 期望的输出 MP4 视频文件路径和文件名
    OUTPUT_VIDEO_FILE = "volleylab_data/output_people/output_BV1Ato6YKEWy_segment_1/tracked_objects_visualized.mp4"
    
    # 输出视频的帧率
    VIDEO_FPS = 25
    # 叠加掩码的透明度 (0.0 表示完全透明，1.0 表示完全不透明)
    OVERLAY_ALPHA = 0.6 

    try:
        create_colored_mask_overlay_video(
            original_frames_dir=ORIGINAL_FRAMES_DIRECTORY,
            masks_base_dir=MASKS_OUTPUT_BASE_DIRECTORY,
            output_video_path=OUTPUT_VIDEO_FILE,
            fps=VIDEO_FPS,
            alpha=OVERLAY_ALPHA
        )
    except Exception as e:
        print(f"发生错误: {e}")
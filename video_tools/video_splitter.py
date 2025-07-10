import cv2
import os

def split_mp4_into_frames(folder_path):
    """
    将指定文件夹内的所有MP4文件分割成帧，并保存到以视频文件名为名的子文件夹中。

    Args:
        folder_path (str): 包含MP4文件的文件夹路径。
    """
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 不存在。")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            video_name = os.path.splitext(filename)[0]  # 获取不带扩展名的视频文件名
            output_folder = os.path.join(folder_path, video_name)

            # 创建输出文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"创建文件夹：{output_folder}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误：无法打开视频文件 '{video_path}'。")
                continue

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频读取完毕

                frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            cap.release()
            print(f"视频 '{filename}' 已分割成 {frame_count} 帧，并保存到 '{output_folder}'。")

# --- 如何使用 ---
if __name__ == "__main__":
    # 将这里替换成你的MP4文件所在的文件夹路径
    input_directory = "volleylab_data/raw"
    split_mp4_into_frames(input_directory)
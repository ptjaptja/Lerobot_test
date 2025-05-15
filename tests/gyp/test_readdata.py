import numpy as np


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


metadata = LeRobotDataset(
    repo_id="ptja/new_dataset",
    root=r"E:\dataset\lerotest5",
    episodes=[1]
)
# 读取第2个episode的内容
print(f"Selected episodes: {metadata.episodes}")
print(f"Number of episodes selected: {metadata.num_episodes}")
print(f"Number of frames selected: {metadata.num_frames}")

# 获取observation.images.cam_high的第10帧图片
camera_key = metadata.meta.camera_keys[0]
frame_index = 9  # 第10帧，索引为9
image = metadata[frame_index][camera_key]
image_hw3 = image.permute(1, 2, 0).numpy()  # 重排为 (H, W, C)
# plt.imshow(image_hw3)

# 显示图片
import matplotlib.pyplot as plt

plt.imshow(image_hw3)
plt.title(f"Episode 1, Frame {frame_index + 1}")
plt.axis('off')
plt.show()


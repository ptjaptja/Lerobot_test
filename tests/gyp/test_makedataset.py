import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


features = {
        "image": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": [
                "channels",
                "height",
                "width",
            ],
        },
        "obs.state":{
            "dtype":"float32",
            "shape":(4,),
            "names":["j1","j2","j3","j4"]
        }
}

# metadata = LeRobotDataset.create(
#     repo_id="ptja/new_dataset",
#     fps=30,
#     root=f"E:\dataset\lerotest4",   # 将创建此目录并写入 meta/info.json 等              # 可选：传入 Robot 实例以自动提取特征
#     features=features,
#     use_videos=True                
# )

metadata = LeRobotDataset(
    repo_id="ptja/new_dataset",
    root=f"E:\dataset\lerotest4"
)
metadata.create_episode_buffer()
ti=np.array([0.0]).astype(np.float32)
for i in range(4):
    
    frame={
        # "timestamp":ti,
        "obs.state":np.random.rand(4).astype(np.float32),
        "image":np.random.rand(3, 96, 128).astype(np.uint8),
        "task":"dummy task2"
        
    }
    metadata.add_frame(frame=frame)
    # ti+=0.1
    #提示错误信息为：timestamps and episode_indices should have the same shape. Found timestamps.shape=(3, 1) and episode_indices.shape=(3,).

metadata.save_episode()
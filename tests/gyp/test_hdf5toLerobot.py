
import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

"""用于将图迈的hdf5（图像压缩）的数据格式转成符合v2.1的lerobot格式，方便后续在gr00t中改造。
需要修改raw_dir,repo_id,root,taskname。其中root的lerotest6这层级文件夹不能提前存在。
taskname修改为描述任务的语言，后续会用于作为language prompt使用
resize参数是为了后续做cosmos的需要，其只能生成特定尺寸的图像，不做cosmos的话此处为False
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=r"E:\dataset\toumai\20241219dataset\test_hdf")
    parser.add_argument("--repo_id", type=str, default="ptja/tumai_sixD")
    parser.add_argument("--root", type=str, default=r"E:\dataset\toumai\20241219dataset\test_lerobot")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use_videos", type=bool, default=True)
    parser.add_argument("--task_name", type=str, default="Left gripper picks up the metal ring,and then handles the metal ring to the right gripper.")
    parser.add_argument("--resize", type=bool, default=False)
    # parser.add_argument("--start_frame", type=int, default=0)
    # parser.add_argument("--end_frame", type=int, default=149)
    return parser.parse_args()



def get_cameras(hdf5_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    rgb_cameras = [key for key in hdf5_data["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118
    return rgb_cameras

#先做整体hdf5集检查，输出路径和其length，以及图像的features
def check_hdf5_dataset(raw_dir):
    compressed_images = "sim" not in raw_dir.name
    
    hdf5_paths = list(raw_dir.glob("*.hdf5"))
    assert len(hdf5_paths) != 0
    hdf5_info = {}
    features = {}
    cam_shape = {}
    action_shape = []
    
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/observations/qpos" in data

            assert data["/action"].ndim == 2
            assert data["/observations/qpos"].ndim == 2

            num_frames = data["/action"].shape[0]
            action_shape.append(data["/action"].shape[1])
            assert num_frames == data["/observations/qpos"].shape[0]            
            
            #将此hdf5_path作为hdf5_info的key，其num_frames作为value
            hdf5_info[hdf5_path] = num_frames
    
    
    #检查图像
    with h5py.File(hdf5_paths[0], "r") as data:
        camera_keys = get_cameras(data)
        for camera in camera_keys:

            if compressed_images:
                assert data[f"/observations/images/{camera}"].ndim == 2
                
            else:
                assert data[f"/observations/images/{camera}"].ndim == 4
                b, h, w, c = data[f"/observations/images/{camera}"].shape
                assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."
            if compressed_images:
                # 假设压缩的图像是以某种方式存储的，例如JPEG压缩
                # 使用cv2.imdecode解压缩图像数据
                compressed_image_data = data[f"/observations/images/{camera}"][0]  # 获取第一帧
                image_array = np.frombuffer(compressed_image_data, dtype=np.uint8)
                decoded_image = cv2.imdecode(image_array, 1)  # 解码为彩色图像
                # image_rgb=cv2.cvtColor(decoded_image,cv2.COLOR_BGR2RGB)
                # plt.imshow(image_rgb)
                # plt.savefig(r"E:\dataset\frame3.png")
                # plt.title(f"Episode 1, Frame {decoded_image + 1}")
                # plt.axis('off')
                # plt.show()
                # cv2.imshow('Image Window', decoded_image)
                # cv2.waitKey(0)
                # 存储解码图像的shape
                
                cam_shape[camera]=decoded_image.shape

    assert all(x == action_shape[0] for x in action_shape)
    features["observation.state"] = {"shape":(action_shape[0],),"dtype":"float32"}
    features["action"] = {"shape":(action_shape[0],),"dtype":"float32"}
    
    #仅top_camera做测试，删掉
    # camera_keys=["cam_high"]
    for camera in camera_keys:
        
        features[f"observation.images.{camera}"] = {"shape":tuple(cam_shape[camera]),"dtype":"video","names": ["height","width","channel"]}
        #todo, 如果需要做cosmos的图像处理，则需要将图像resize成特定尺寸
        # if not args.resize:
        #     features[f"observation.images.{camera}"] = {"shape":tuple([704,960,3]),"dtype":"video","names": ["height","width","channel"]}
    return hdf5_info,features

# def load_frame_from_

def main():
    args = parse_args()
    
    raw_dir=Path(args.raw_dir)
    hdf5_info,features=check_hdf5_dataset(raw_dir)
    print(hdf5_info)
    print(features)
    
    
    metadata = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,   # 将创建此目录并写入 meta/info.json 等              # 可选：传入 Robot 实例以自动提取特征
        features=features,
        use_videos=args.use_videos                
    )

    for hdf5_path,num_frames in tqdm.tqdm(hdf5_info.items(),desc="Processing HDF5 files"):
        metadata.create_episode_buffer()
        with h5py.File(hdf5_path, "r") as data:
            
            # for i in tqdm.tqdm(range(args.start_frame,args.end_frame),desc="Processing frames",leave=False):
            for i in tqdm.tqdm(range(num_frames),desc="Processing frames",leave=False):
            
                frame={}
                camera_keys = get_cameras(data)
                #此处可选择只做某个camera的图像处理
                # camera_keys=["cam_high"]
                
                for camera in camera_keys:
                    compressed_image_data = data[f"/observations/images/{camera}"][i]  # 获取第i帧
                    image_array = np.frombuffer(compressed_image_data, dtype=np.uint8)
                    decoded_image = cv2.imdecode(image_array, 1)  # 解码为彩色图像
                    image_rgb=cv2.cvtColor(decoded_image,cv2.COLOR_BGR2RGB)
                    if args.resize:
                        target_size = (960, 704)
                        image_rgb=cv2.resize(image_rgb,target_size, interpolation=cv2.INTER_AREA)
                    frame[f"observation.images.{camera}"] = image_rgb
                
                #转换成float32
                frame["observation.state"]=data["/observations/qpos"][i].astype(np.float32)
                frame["action"]=data["/action"][i].astype(np.float32)
                frame["task"]=args.task_name
                metadata.add_frame(frame=frame)
        
        metadata.save_episode()
    
    print("done")



if __name__ == "__main__":
    main()


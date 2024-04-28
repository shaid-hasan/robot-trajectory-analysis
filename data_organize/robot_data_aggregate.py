import glob
import os
import ast
import time
import csv
import pandas as pd
import numpy as np
import shutil

def get_fileName(_episode_path):
    robot_pose_path = (_episode_path + "/data.pkl" 
                   if "interface_3" in _episode_path 
                   else glob.glob(_episode_path + "/*.csv")[0])
    realsense_path = _episode_path + "/realsense_timestamps.pkl"
    kinect1_path = _episode_path + "/kinect1_timestamps.pkl"
    kinect2_path = _episode_path + "/kinect2_timestamps.pkl"

    if all(map(os.path.exists, [robot_pose_path, realsense_path, kinect1_path, kinect2_path])):
        return robot_pose_path, realsense_path, kinect1_path, kinect2_path
    else:
        return None, None, None, None

data_dir = "/project/CollabRoboGroup/datasets/franka_multimodal_teleop/task_8"
for _taskdir in sorted(glob.glob(data_dir)):
    interface_dir = _taskdir + "/interface_*"
    for _interface in sorted(glob.glob(interface_dir)):
        episode_dir = _interface + "/episode_*"
        for _episode_path in sorted(glob.glob(episode_dir), key=lambda x: int(x.split("_")[-1])):

            # print("episode_path:",_episode_path)
            robot_pose_path, _, _, _ = get_fileName(_episode_path)
            print("robot_pose_path:",robot_pose_path)

            if robot_pose_path and os.path.exists(robot_pose_path):
                new_robot_episode_path = _episode_path.replace("datasets", "qmz9mg").replace("franka_multimodal_teleop", "franka_teleop_robot_data")
                # print("new_robot_episode_path:", new_robot_episode_path)
                shutil.copy(robot_pose_path, new_robot_episode_path)
                print ("File copied successfully!")

            else:
                print ("No file!")

            # break
        # break
    # break

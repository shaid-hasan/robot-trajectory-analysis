import glob
import os
from frame_match import *
import glob
from natsort import natsorted
from PIL import Image
import threading
import cv2
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})

print("All packages installed!")

data_dir = "/project/CollabRoboGroup/datasets/franka_multimodal_teleop/task_*"

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


def read_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)     
        
    return np.asarray(data).reshape(-1,1)


def read_robot_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)
        timestamps = np.array(data['timestamps']).reshape(-1,1)
        observations = data['observations']
        return timestamps, observations

def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    data = np.asarray(data)
    return (data)[1:, 0].reshape(-1,1), data


def make_video(file_path, corr_indices, cam_identifier):
    
    # Path for writing new images 
    color_dir = file_path + "/synchronized" + cam_identifier + "_color"
    depth_dir = file_path + "/synchronized" + cam_identifier + "_depth"
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path_color = file_path + cam_identifier + "_color"
    file_path_depth = file_path + cam_identifier + "_depth"


    image_path_color = file_path_color + "/*.png"
    count = 0
    for i, color_img in enumerate(natsorted(glob.glob(image_path_color))):
        if i in corr_indices:
            cv2_image = cv2.imread(color_img)
            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f'{color_dir}/{count}.png')
            count +=1

            if count==10:
                break

    image_path_depth = file_path_depth + "/*.png"
    count = 0
    for i, depth_img in enumerate(natsorted(glob.glob(image_path_depth))):
        if i in corr_indices:
            cv2_depth_image = cv2.imread(depth_img, cv2.IMREAD_GRAYSCALE)
            colored_depth_image = cv2.applyColorMap(cv2_depth_image, cv2.COLORMAP_BONE)
            pil_colored_depth_image = Image.fromarray(cv2.cvtColor(colored_depth_image, cv2.COLOR_BGR2RGB))
            pil_colored_depth_image.save(f'{depth_dir}/{count}_depth.png')
            count +=1

            if count==10:
                break
            

def write_skeleton(corr_indices, robot_pose_data, robot_pose_path, file_path):
    print (len(corr_indices))
    write_folder = file_path + "/synchronized" + "/robot_data.csv"
    
    new_robot_pose = list()
    df = pd.read_csv(robot_pose_path)
    # print ("old df ",len(df))
    # print("df:",df)
    df = df.iloc[:len(corr_indices)]

    df = df.reset_index(drop=True)
    # print ("new df ",len(df))
    # print("df:",df)
    df.to_csv(write_folder)


for _taskdir in sorted(glob.glob(data_dir)):
    interface_dir = _taskdir + "/interface_1"
    for _interface in sorted(glob.glob(interface_dir)):
        episode_dir = _interface + "/episode_*"
        for _episode_path in sorted(glob.glob(episode_dir), key=lambda x: int(x.split("_")[-1])):

            robot_pose_path, realsense_path, kinect1_path, kinect2_path = get_fileName(_episode_path)

            if os.path.exists(realsense_path) and os.path.exists(kinect1_path) and os.path.exists(kinect2_path) and os.path.exists(robot_pose_path):
                    print ("reading pickle files")
                    realsense_timestamps = read_pickle(realsense_path)
                    kinect1_timestamps = read_pickle(kinect1_path)
                    kinect2_timestamps = read_pickle(kinect2_path)

                    print("robot_pose_path:",robot_pose_path)

                    if "interface_3" in _episode_path:
                        robot_timestamps, robot_pose_data = read_robot_pickle(robot_pose_path)
                    else:
                        robot_timestamps, robot_pose_data = read_csv(robot_pose_path)

                    query_timestamp = robot_timestamps
                    if len(robot_timestamps)>1:
                        query_timestamp, corr_indices_k1_k1 = loop_queryarray(query_timestamp, kinect1_timestamps, denom=1)
                        query_timestamp, corr_indices_k1_k2 = loop_queryarray(query_timestamp, kinect2_timestamps, denom=1)
                        query_timestamp, corr_indices_k1_w = loop_queryarray(query_timestamp, realsense_timestamps, denom=1)
                        _, corr_indices_k1_r = loop_queryarray(query_timestamp, robot_timestamps, denom=1)

                        print ("timestamps for all the streans ", len(set(corr_indices_k1_w)),len(set(corr_indices_k1_k2)),len(set(corr_indices_k1_r)), len(set(corr_indices_k1_k1)))    

                        # make_video(_episode_path, corr_indices_k1_w, "/rs")
                        # make_video(_episode_path, corr_indices_k1_k1, "/kinect1")
                        # make_video(_episode_path, corr_indices_k1_k2, "/kinect2")

                        # write_skeleton(list(set(corr_indices_k1_r)), robot_pose_data, robot_pose_path, _episode_path) 

                        

            # break
        break
    break


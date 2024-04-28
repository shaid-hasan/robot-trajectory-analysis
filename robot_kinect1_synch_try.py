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
# np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})
from scipy.interpolate import interp1d
import csv

# print("All packages installed!")

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
        np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})
    return np.asarray(data).reshape(-1,1)


def read_robot_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)
        # timestamps = data['timestamps']
        timestamps = np.array(data['timestamps']).reshape(-1,1)
        np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})

        observations = data['observations']
        print(type(observations))
        print(type(observations[0]))
        print("observations:",observations[0])
        print("observations-keys:",list(observations[0].keys()))
        #  print("observations-keys['robot_state']:",list(observations[0]['robot_state'].keys()))

        return timestamps

# def read_csv(csv_file):
#     data = pd.read_csv(csv_file)
#     data = np.asarray(data)
#     return (data)[1:, 0].reshape(-1,1), data

def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    # Convert the timestamp column to float64
    timestamps = data.iloc[1:, 0].astype(np.float64).values.reshape(-1, 1)
    return timestamps, data.values


def make_video(file_path, corr_indices, cam_identifier):
    print("corr_indices:",corr_indices)
    count = 0
    # Path for writing new images 
    write_folder = file_path + "/synchronized" + cam_identifier
    os.makedirs(write_folder, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path = file_path + cam_identifier
    camera = file_path.split("/")[-1]
    # print("camera:",camera)

    image_path = file_path + "/*.png"

    for i, image_path in enumerate(natsorted(glob.glob(image_path))):
        
        if i in corr_indices:
            print("i:",i)
            print("image_path:",image_path)
            cv2_image = cv2.imread(image_path)
            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f'{write_folder}/{count}.png')
            count +=1

            if count == 50:
                break
            
    return

def write_skeleton(corr_indices, robot_pose_data, robot_pose_path, file_path):
    print (len(corr_indices))
    write_folder = file_path + "synchronized" + "/robot_data.csv"
    
    new_robot_pose = list()
    df = pd.read_csv(robot_pose_path)
    # print ("old df ",len(df))
    df = df.iloc[:len(corr_indices)]

    df = df.reset_index(drop=True)
    # print ("new df ",len(df))
    df.to_csv(write_folder)


for _taskdir in sorted(glob.glob(data_dir)):
    interface_dir = _taskdir + "/interface_*"
    for _interface in sorted(glob.glob(interface_dir)):
        episode_dir = _interface + "/episode_*"
        for _episode_path in sorted(glob.glob(episode_dir), key=lambda x: int(x.split("_")[-1])):

            robot_pose_path, realsense_path, kinect1_path, kinect2_path = get_fileName(_episode_path)
            # print("robot_pose_path",robot_pose_path)
            # print("realsense_path",realsense_path)
            # print("kinect1_path",kinect1_path)
            # print("kinect2_path",kinect2_path)

            if os.path.exists(realsense_path) and os.path.exists(kinect1_path) and os.path.exists(kinect2_path) and os.path.exists(robot_pose_path):
                    realsense_timestamps = read_pickle(realsense_path)
                    kinect1_timestamps = read_pickle(kinect1_path)
                    kinect2_timestamps = read_pickle(kinect2_path)
                    robot_timestamps, robot_pose_data = read_csv(robot_pose_path)
                    query_timestamp = robot_timestamps

                    print("Kinect1.shape before synch:", kinect1_timestamps.shape)
                    print("Kinect2.shape before synch:", kinect2_timestamps.shape)
                    print("realsense.shape before synch:", realsense_timestamps.shape)
                    print("Robot.shape before synch:", robot_timestamps.shape)

                    robot_interp = interp1d(robot_timestamps.flatten(), np.arange(len(robot_timestamps)), fill_value="extrapolate")
                    corresponding_indices = np.round(robot_interp(kinect1_timestamps.flatten())).astype(int)
                    valid_indices = (corresponding_indices >= 0) & (corresponding_indices < len(robot_timestamps))
                    robot_timestamps_synch = robot_timestamps[corresponding_indices[valid_indices]]
                    kinect1_timestamps_synch = kinect1_timestamps[valid_indices]

                    with open('robot_came_synch_check.csv', mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Kinect1 Timestamps Before', 'Robot Timestamps Before', 'Kinect1 Timestamps After', 'Robot Timestamps After'])
                        max_len = max(len(kinect1_timestamps), len(robot_timestamps), len(kinect1_timestamps_synch), len(robot_timestamps_synch))
                        for i in range(max_len):
                            kinect1_before = kinect1_timestamps[i][0] if i < len(kinect1_timestamps) else None
                            robot_before = robot_timestamps[i][0] if i < len(robot_timestamps) else None
                            kinect1_after = kinect1_timestamps_synch[i][0] if i < len(kinect1_timestamps_synch) else None
                            robot_after = robot_timestamps_synch[i][0] if i < len(robot_timestamps_synch) else None
                            writer.writerow([kinect1_before, robot_before, kinect1_after, robot_after])


                    print("Kinect1.shape after synch:", kinect1_timestamps_synch.shape)
                    print("Robot.shape after synch:", robot_timestamps_synch.shape)

                        
            break
        break
    break


import glob
import os
import ast
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})

# 1. Sweeping object into dustpan
# 2. Drawer Closing
# 3. Opening a drawer
# 4. Pouring from glass 
# 5. Cube Stacking 
# 6. Cup Stacking  
# 7. Cloth folding
# 8. Pressing a button 

data_dir = "/project/CollabRoboGroup/datasets/franka_multimodal_teleop/task_1"

def csv_write(array, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)

def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    data = np.asarray(df)
    xyz_values = []

    for index, row in df.iterrows():
        eef_pos_dict = ast.literal_eval(row['eef_pos'])
        xyz_values.append([eef_pos_dict['x'], eef_pos_dict['y'], eef_pos_dict['z']])
    xyz_pose = np.array(xyz_values)

    return (data)[1:, 0].reshape(-1,1), xyz_pose

def read_robot_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)
        timestamps = np.array(data['timestamps']).reshape(-1,1)
        observations = data['observations']
        return timestamps, observations

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


def remove_consecutive_repeated_elements(array):
    mask = np.concatenate(([True], np.any(array[1:] != array[:-1], axis=1)))
    filtered_array = array[mask]
    return filtered_array

def downsample_array(array, num_samples):
    interval = array.shape[0] // num_samples
    downsampled_array = array[::interval]
    return downsampled_array

for _taskdir in sorted(glob.glob(data_dir)):
    interface_dir = _taskdir + "/interface_2"
    for _interface in sorted(glob.glob(interface_dir)):
        episode_dir = _interface + "/episode_*"
        for _episode_path in sorted(glob.glob(episode_dir), key=lambda x: int(x.split("_")[-1])):

            print("episode_path:",_episode_path)
            robot_pose_path, _, _, _ = get_fileName(_episode_path)

            if "interface_3" in _episode_path:
                robot_timestamps, xyz_pose = read_robot_pickle(robot_pose_path)
            else:
                robot_timestamps, xyz_pose = read_csv(robot_pose_path)

            print("original:", xyz_pose.shape)
            xyz_pose=remove_consecutive_repeated_elements(xyz_pose)
            print("filtered:", xyz_pose.shape)
            xyz_pose = downsample_array(xyz_pose, num_samples=5)
            print("sampled:", xyz_pose.shape)

            # x_samples = []
            # y_samples = []
            # z_samples = []

            # for i in range(0, xyz_pose.shape[0], interval):
            #     x_samples.append(xyz_pose[i][0])
            #     y_samples.append(xyz_pose[i][1])
            #     z_samples.append(xyz_pose[i][2])
            



            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.view_init(elev=30, azim=30) # task-1,2,3
            # # ax.view_init(elev=30, azim=60) # task-4
            # ax.view_init(elev=15, azim=20) # task-5,6
            # # ax.view_init(elev=15, azim=75) # task-7,8




            # ax.scatter(x_samples[1:-1], y_samples[1:-1], z_samples[1:-1], c='b', s=15,marker='o')
            # ax.plot(x_samples, y_samples, z_samples, c='black')
            # ax.scatter(x_samples[0], y_samples[0], z_samples[0], c='g', marker='v', s=200, label='Start')
            # ax.scatter(x_samples[-1], y_samples[-1], z_samples[-1], c='red', marker='X', s=200, label='End')

            # ax.set_xlim(0.2, 0.75)
            # ax.set_ylim(-0.4, 0.3)
            # ax.set_zlim(0, 0.6)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_title('Regularly Sampled 3D Points')

            # plt.savefig('sampled_3d_points.png')
            # plt.show()

            # time.sleep(2.0)

            break
            
        break
    break
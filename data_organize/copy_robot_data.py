import os
import shutil
import time

## 3D Mouse and Joystick
# user_setA = [102, 103, 104, 105, 107, 108, 110, 111, 112, 113, 116, 117, 119, 120, 121]
# user_setB = [122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]

## VR
# user_setA = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 21]
user_setB = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
# user_setB = [22]

# task_setA = [1, 3, 5, 7]
task_setB = [2, 4, 6, 8]
interface_numbers = [3]

for user_id in user_setB:
    for task in task_setB:
        for interface_number in interface_numbers:

            dest_path = f'/project/CollabRoboGroup/datasets/franka_multimodal_teleop/task_{task}/interface_{interface_number}/episode_{user_setB.index(user_id) + 1}'
            os.makedirs(dest_path, exist_ok=True)
            
            start_time = time.time()
            # source_path = f'/project/CollabRoboGroup/qmz9mg/VR_Interface_data/P_{user_id}/Task_{task}/Interface_{interface_number}/Trial_1'
            # source_path = '/project/CollabRoboGroup/qmz9mg/VR_Interface_data/Participant_2/Task_1/Interaface_3/Trial_1/'
            source_path = f'/project/CollabRoboGroup/qmz9mg/VR_Interface_data/Participant_{user_id}/Task_{task}/Interaface_3/Trial_1/'

            if os.listdir(source_path):
                try:
                    for root, dirs, files in os.walk(source_path):
                        relative_path = os.path.relpath(root, source_path)
                        dest_dir = os.path.join(dest_path, relative_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        for file in files:
                            shutil.copy(os.path.join(root, file), dest_dir)
                except Exception as e:
                    print("An error occurred:", e)

                print("Successfuly copied: User ID:", user_id, "Task Number:", task, "Interface Number:", interface_number, "Time:", time.time() - start_time)
            else:
                print("No files found in the source directory.")
                print("User ID:", user_id, "Task Number:", task, "Interface Number:", interface_number, "Time:", time.time() - start_time)

            
import os
from collections import defaultdict

directory = "/project/CollabRoboGroup/qmz9mg/RGB_Data/P_24/Task_2/Interface_3/Trial_1/rs_depth/"

files_by_id = defaultdict(list)

for filename in os.listdir(directory):
    if filename.startswith("depth_frame_"):
        parts = filename.split("_")
        id = parts[2]  
        timestamp = "_".join(parts[3:]) 
        files_by_id[id].append((filename, timestamp))

for id, file_list in files_by_id.items():
    if len(file_list) == 1:
        os.remove(os.path.join(directory, file_list[0][0]))
        print(f"Deleted file: {file_list[0][0]}")
    elif len(file_list) > 1:
        file_list.sort(key=lambda x: x[1], reverse=True)
        os.remove(os.path.join(directory, file_list[1][0]))
        print(f"Deleted file: {file_list[1][0]}")

print("Deletion process completed.")
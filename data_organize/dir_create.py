import os

dataset_root_dir = '/project/CollabRoboGroup/qmz9mg/franka_teleop_robot_data'

# Create task directories
for task_id in range(1, 9):
    task_dir = os.path.join(dataset_root_dir, f'task_{task_id}')
    os.makedirs(task_dir, exist_ok=True)
    
    # Create interface directories
    for interface_id in range(1, 4):
        interface_dir = os.path.join(task_dir, f'interface_{interface_id}')
        os.makedirs(interface_dir, exist_ok=True)
        
        # Create episode directories
        for episode_number in range(1, 16):
            episode_dir = os.path.join(interface_dir, f'episode_{episode_number}')
            os.makedirs(episode_dir, exist_ok=True)

print("Directory structure created successfully!")

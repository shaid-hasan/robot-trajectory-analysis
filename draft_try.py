import numpy as np

# Generate a 20x3 array of random float numbers
random_array = np.random.rand(20, 3)

# Introduce consecutive same elements in some rows
# Let's say we want to repeat the first element in rows 2, 5, and 10
random_array[1:3, :] = random_array[0, :]
random_array[7:10, :] = random_array[3, :]
random_array[16:19, :] = random_array[19, :]

# Display the array
print("original:\n",random_array)
print("original.shape:\n",random_array.shape)

def remove_consecutive_repeated_elements(array):
    mask = np.concatenate(([True], np.any(array[1:] != array[:-1], axis=1)))
    filtered_array = array[mask]
    return filtered_array

filtered_array=remove_consecutive_repeated_elements(random_array)
print("filtered:\n",filtered_array)
print("filtered.shape:\n",filtered_array.shape)
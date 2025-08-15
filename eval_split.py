import os

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

data_path = '/data3/mutian/lasa_cad_depth'    
scene_id_list = sorted(os.listdir(data_path))

num_gpus = 4

# Split the list into num_gpus splits
splits = list(split_list(scene_id_list, num_gpus))

# Write each split to a separate text file
for i, split_list in enumerate(splits):
    with open(f'split_{i}.txt', 'w') as file:
        for item in split_list:
            file.write(f"{item}\n")
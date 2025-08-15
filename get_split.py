# generate the split from scene_id to scene_id/frame_id

# Read the lines from the file into a list
with open('val.lst', 'r') as file:
    lines = file.readlines()

# Extract unique strings before '/' and store them in a set to ensure uniqueness
unique_strings = set()
for line in lines:
    string_before_slash = line.split('/')[0]
    unique_strings.add(string_before_slash)

# Convert the set of unique strings back to a list
unique_strings_list = list(unique_strings)

# Print the list of unique strings
print(len(unique_strings_list))

# Write the unique strings to a new file, each on a new line
with open('val_scene.lst', 'w') as file:
    for item in unique_strings_list:
        file.write(item + '\n')
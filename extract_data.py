import os
import json

# Set your folder path containing JSON files
folder_path = 'IVF (Fertility) VoxBot'

# Initialize a list to hold contents
all_data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)  # Append each JSON object

# Write the combined data into a single JSON file
output_path = os.path.join(folder_path, 'merged.json')
with open('./', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4)

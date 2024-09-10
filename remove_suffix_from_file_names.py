import os

def rename_csv_files(directory_path):
    # List all files in the given directory
    for filename in os.listdir(directory_path):
        # Check if the file ends with "_zeroed.csv"
        if filename.endswith("_zeroed.csv"):
            # Create the new filename by removing "_zeroed" part
            new_filename = filename.replace("_zeroed", "")
            # Construct the full old and new file paths
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

# Provide the path to the directory containing the CSV files
directory_path = 'your_directory_path_here'

# Call the function to rename CSV files
rename_csv_files(directory_path)

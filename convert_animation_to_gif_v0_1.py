import moviepy.editor as mp
import os
import glob
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

def convert_video_to_gif(input_path, output_path):
    # Load the video
    clip = mp.VideoFileClip(input_path)
    
    # Convert to GIF
    clip.write_gif(output_path, fps=10)  # You can adjust the fps value

def convert_all_videos_in_folder(folder_path):
    # Find all MP4 and AVI files in the folder
    video_files = glob.glob(os.path.join(folder_path, '*.mp4')) + glob.glob(os.path.join(folder_path, '*.avi'))
    
    for video_file in video_files:
        # Create the output file path
        output_file = os.path.splitext(video_file)[0] + '.gif'
        
        print(f"Converting {video_file} to {output_file}")
        convert_video_to_gif(video_file, output_file)
        print(f"Conversion complete: {output_file}")

def select_folder():
    app = QApplication([])
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
    return folder_path

if __name__ == "__main__":
    folder_path = select_folder()
    
    if folder_path:
        convert_all_videos_in_folder(folder_path)
        print("All conversions complete.")
    else:
        print("No folder selected.")

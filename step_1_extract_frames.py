import os
import imageio
from moviepy.editor import VideoFileClip

# Input video file
input_video = "test.mp4"

# Output folder to save frames
output_folder = "test_images"
cartonize_path = "cartoonized_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    # Check if the output folder is not empty, and if it is, remove its contents
    existing_files = os.listdir(output_folder)
    if len(existing_files) > 0:
        for file_name in existing_files:
            file_path = os.path.join(output_folder, file_name)
            os.remove(file_path)

if not os.path.exists(cartonize_path):
    os.makedirs(cartonize_path)
else:
    # Check if the output folder is not empty, and if it is, remove its contents
    existing_files = os.listdir(cartonize_path)
    if len(existing_files) > 0:
        for file_name in existing_files:
            file_path = os.path.join(cartonize_path, file_name)
            os.remove(file_path)


# Frame rate (frames per second)
frame_rate = 5

# Load the video
video_clip = VideoFileClip(input_video)

# Iterate through the video frames and save them as numbered PNG files
for i, frame in enumerate(video_clip.iter_frames(fps=frame_rate), start=1):
    # Save each frame as a PNG file with a 3-digit numbering (e.g., 001.png, 002.png, ...)
    filename = f"{i:03d}.png"
    frame_path = os.path.join(output_folder, filename)
    imageio.imsave(frame_path, frame)

# Close the video clip
video_clip.reader.close()

import os
import subprocess

# Input folder containing processed frames
input_folder = "cartoonized_images"

# Output video file
output_video = "processed_test_5.mp4"

# Frame rate (frames per second)
frame_rate = 5

# Run FFmpeg to create the video from processed frames
cmd = [
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", os.path.join(input_folder, "%03d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_video
]

subprocess.run(cmd)

print(f"Video '{output_video}' created successfully.")
import os

import imageio
from moviepy.editor import VideoFileClip


def extract_frames(video_path, temp_dir, frame_rate=5):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        # Clear the temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
    video_clip = VideoFileClip(video_path)
    frame_paths = []
    for i, frame in enumerate(video_clip.iter_frames(fps=frame_rate), start=1):
        filename = f"{i:03d}.png"
        frame_path = os.path.join(temp_dir, filename)
        imageio.imsave(frame_path, frame)
        frame_paths.append(frame_path)
    video_clip.reader.close()
    return frame_paths


def create_video_from_frames(frame_folder, output_video, frame_rate=5):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    # Use ffmpeg to compile frames into a video
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file if exists
        "-framerate", str(frame_rate),
        "-i", os.path.join(frame_folder, "%03d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(cmd, check=True)


def cleanup_temp(temp_dir):
    # Remove all files in the temporary directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))

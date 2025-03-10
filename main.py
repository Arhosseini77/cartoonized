import argparse
import os

import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from model import CartoonizerModel
from utils import utils_image, utils_video


def process_image(input_path, output_path, model):
    # Load, preprocess, and cartoonize image
    image = utils_image.load_image(input_path)
    processed_image = utils_image.preprocess_image(image)
    cartoonized = model.cartoonize(processed_image)
    # Denormalize result to [0,255]
    cartoonized = (cartoonized + 1) * 127.5
    cartoonized = np.clip(cartoonized, 0, 255).astype(np.uint8)
    utils_image.save_image(cartoonized, output_path)
    print(f"Cartoonized image saved to {output_path}")


def process_video(input_video, output_video, temp_dir, model, fallback_frame_rate):
    # Open video to check duration and get fps
    clip = VideoFileClip(input_video)
    if clip.duration > 63:
        print("Error: Please use a video shorter than 1 minutes.")
        clip.reader.close()
        return
    input_fps = clip.fps if clip.fps is not None else fallback_frame_rate
    clip.reader.close()

    # Extract frames from video using the input video's fps
    frame_paths = utils_video.extract_frames(input_video, temp_dir, frame_rate=input_fps)

    # Process each frame with a progress bar
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        image = utils_image.load_image(frame_path)
        processed_image = utils_image.preprocess_image(image)
        cartoonized = model.cartoonize(processed_image)
        cartoonized = (cartoonized + 1) * 127.5
        cartoonized = np.clip(cartoonized, 0, 255).astype(np.uint8)
        utils_image.save_image(cartoonized, frame_path)

    # Create video from processed frames using the input video's fps
    utils_video.create_video_from_frames(temp_dir, output_video, frame_rate=input_fps)

    # Cleanup temporary frames
    utils_video.cleanup_temp(temp_dir)
    print(f"Cartoonized video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Cartoonize Images or Videos")
    parser.add_argument('--mode', type=str, choices=['image', 'video'], required=True,
                        help="Processing mode: image or video")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input image or video file")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the output file (or directory for images)")
    parser.add_argument('--model_path', type=str, default='weights',
                        help="Directory containing the pre-trained model weights")
    # fallback frame rate in case the input video's fps is not available
    parser.add_argument('--frame_rate', type=int, default=5,
                        help="Fallback frame rate for video processing if input fps is not available")
    args = parser.parse_args()

    # Disable eager execution (for TF1 compatibility)
    tf.compat.v1.disable_eager_execution()

    # Initialize and load the model
    model_instance = CartoonizerModel(model_path=args.model_path)
    model_instance.load_model()

    if args.mode == 'image':
        # If output is a directory, use input file name
        if os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.basename(args.input)
            output_path = os.path.join(args.output, base_name)
        else:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            output_path = args.output
        process_image(args.input, output_path, model_instance)
    elif args.mode == 'video':
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # Use a temporary folder for frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        process_video(args.input, args.output, temp_dir, model_instance, args.frame_rate)


if __name__ == '__main__':
    main()

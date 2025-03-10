# Cartoonization Image and Video

This project provides a clean, modular implementation for cartoonizing images and videos. It is based on the paper [CVPR2020] Learning to Cartoonize Using White-box Cartoon Representations.

## Project Structure

```
project/
├── model/
│   ├── __init__.py
│   └── model.py         # Network architecture and guided filter implementation
├── utils/
│   ├── __init__.py
│   ├── utils_image.py   # Image processing utilities
│   └── utils_video.py   # Video processing utilities
├── main.py              # Main script to choose between image and video cartoonization
├── saved_models/        # Directory containing pre-trained models
└── README.md
```

## Installation

create conda env with **python 3.7**
````
conda create -n cartoon python=3.7
````
install tensorflow 1.15 
```
conda install -c anaconda cudatoolkit=10.0
conda install -c anaconda cudnn=7.6.5
pip install tensorflow-gpu==1.15
```

install requirements
````
conda env update --file cartoon_env.yml --prune
````

install Numpy 
````
pip install numpy==1.19.5
````

## Inference

### For Images

To cartoonize a single image:
```bash
python main.py --mode image --input path/to/input_image.jpg --output path/to/output_image.jpg --model_path saved_models
```

If the output path is a directory, the output file will use the input image’s filename.

### For Videos

To cartoonize a video:
```bash
python main.py --mode video --input path/to/input_video.mp4 --output path/to/output_video.mp4 --model_path saved_models --frame_rate 5
```

*Video processing will extract frames into a temporary folder (`temp_frames`), process them, create the final video, and then clean up the temporary files automatically.*

## References

- [Project Page](https://systemerrorwang.github.io/White-box-Cartoonization/)
- [Paper](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf)
```
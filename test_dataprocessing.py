# 1. detection de visage torch 
# 2. transformation (rotation, translation, scaling, random cropping & resizing, color transformation, elastic deformations or grid distortions)
# 3. Resize t& rescalling: resize, prevent aspect ration distortion, normalization or standardization (trap pixel value in range)

import os
from torchvision.io import read_video 
import av
import torch 
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import pickle
from torchvision.transforms import ToTensor

# import raw data & extract frames from video
def extract_frames(video_path, output_dir, interval=1):

    container = av.open(video_path)
    os.makedirs("framed_videos", exist_ok=True)

    frame_index = 0
    # Iterate over the video frames
    for frame in container.decode(video=0):
        # Check if the frame index is divisible by the interval
        if frame_index % interval == 0:
            # Convert the frame to a PIL Image
            image = frame.to_image()

            # Save the frame as an image file
            image.save(os.path.join(output_dir, f"frame_{frame_index}.jpg"))

        # Increment the frame index
        frame_index += 1


video_path = "../input/automathon-deepfake/dataset/experimental_dataset" # change this
output_dir = "framed_videos"


# resize for validation
def resize_images(input_dir, output_dir, size=(256, 256)):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in image_files:
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Resize the image and save it
        resized_image = image.resize(size)
        resized_image.save(os.path.join(output_dir, filename))

input_dir = "framed_videos"
output_dir = "resized_test_images"
resize_images(input_dir, output_dir, size=(256, 256))


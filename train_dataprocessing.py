# 1. detection de visage torch 
# 2. transformation (rotation, translation, scaling, random cropping & resizing, color transformation, elastic deformations or grid distortions)
# 3. Resize t& rescalling: resize, prevent aspect ration distortion, normalization or standardization (trap pixel value in range)

import os
from torchvision.io import read_video 
import av
import torch 
from PIL import Image
from PIL import ImageFilter

# import raw data & extract frames from video
def extract_frames(video_path, output_dir, interval=10):

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


video_path = "../input/automathon-deepfake/dataset/test_dataset" # change this
output_dir = "framed_videos"

#extract_frames(video_path, output_dir)

 # !!!! resize les images

# data augmentation manipulate images
def augmentation(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all image files in the input directory
    image_files = os.listdir(input_dir)

    for filename in image_files:
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Rotate the image by 45 degrees and save it
        rotated_image_45 = image.rotate(45)
        rotated_image_45.save(os.path.join(output_dir, f"{filename}_rotated_45.jpg"))

        # Rotate the image by 180 degrees and save it
        rotated_image_180 = image.rotate(180)
        rotated_image_180.save(os.path.join(output_dir, f"{filename}_rotated_180.jpg"))

        # Translate the image by 5 pixels to the right and 5 pixels down and save it
        translated_image = image.transform((image.width, image.height), Image.AFFINE, (1, 0, 5, 0, 1, 5))
        translated_image.save(os.path.join(output_dir, f"{filename}_translated.jpg"))

        # Apply horizontal flip and save it
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_image.save(os.path.join(output_dir, f"{filename}_flipped.jpg"))

        # Apply Gaussian noise and save it
        noisy_image = image.filter(ImageFilter.GaussianBlur(radius=2))  # Adjust radius as needed
        noisy_image.save(os.path.join(output_dir, f"{filename}_noisy.jpg"))


fake = False

if fake:
    input_dir = "light_fake"
    output_dir = "augmented_images/fake_augmented"
    augmentation(input_dir, output_dir)
else:
    input_dir = "light_real"
    output_dir = "augmented_images/real_augmented"
    augmentation(input_dir, output_dir)


# a function specifically for treating training dataset
'''def aggregted(input_dir, output_dir, size=(256, 256)):
    extract_frames(video_path, output_dir, interval=1)
    augmentation(input_dir, output_dir)
'''

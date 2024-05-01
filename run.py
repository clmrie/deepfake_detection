
#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from torchvision.io.video import re
from tqdm import tqdm
import csv
import timm
import wandb
import random
import av

from PIL import Image
import torchvision.transforms.v2 as transforms


# face detection
#from facenet_pytorch import MTCNN



# UTILITIES

def extract_frames(video_path, nb_frames=10, delta=1, timeit=False):
    # use time to measure the time it takes to resize a video
    t1 = time.time()
    reader = io.VideoReader(video_path)
    # take 10 frames uniformly sampled from the video
    frames = []
    for i in range(nb_frames):
        reader.seek(delta)
        frame = next(reader)
        frames.append(frame['data'])
    t2 = time.time()     
    video = torch.stack(frames)
    if timeit:
        print(f"read: {t2-t1}")
    return video

def smart_resize(data, size): # kudos louis
    # Prends un tensor de shape [...,C,H,W] et le resize en [...C,size,size]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]

    if full_height > full_width:
        alt_height = size
        alt_width = int(full_width / (full_height / size))
    elif full_height < full_width:
        alt_height = int(full_height / (full_width / size))
        alt_width = size
    else:
        alt_height = size
        alt_width = size
    tr = transforms.Compose([
        transforms.Resize((alt_height, alt_width)),
        transforms.CenterCrop(size)
    ])
    return tr(data)

def resize_data(data, new_height, new_width, x=0, y=0, height=None, width=None):
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]
    height = full_height - y if height is None else height
    width = full_width -x if width is None else width

    ratio = new_height/new_width
    if height/width > ratio:
        expand_height = height
        expand_width = int(height / ratio)
    elif height/width < ratio:
        expand_height = int(width * ratio)
        expand_width = width
    else:
        expand_height = height
        expand_width = width
    tr = transforms.Compose([
        transforms.CenterCrop((expand_height, expand_width)),
        transforms.Resize((new_height, new_width))
    ])
    x = data[...,y:min(y+height, full_height), x:min(x+width, full_width)].clone()
    return tr(x)


# SETUP DATASET

dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
nb_frames = 10

## MAKE RESIZED DATASET
resized_dir = os.path.join(dataset_dir, "resized_dataset")
"""
create_small_dataset = False
errors = []
if not os.path.exists(resized_dir) or create_small_dataset:
    os.mkdir(resized_dir)
    os.mkdir(os.path.join(resized_dir, "train_dataset"))
    os.mkdir(os.path.join(resized_dir, "test_dataset"))
    os.mkdir(os.path.join(resized_dir, "experimental_dataset"))
    train_files = [f for f in os.listdir(os.path.join(dataset_dir, "train_dataset")) if f.endswith('.mp4')]
    test_files = [f for f in os.listdir(os.path.join(dataset_dir, "test_dataset")) if f.endswith('.mp4')]
    experimental_files = [f for f in os.listdir(os.path.join(dataset_dir, "experimental_dataset")) if f.endswith('.mp4')]
    def resize(in_video_path, out_video_path, nb_frames=10):
        video = extract_frames(in_video_path, nb_frames=nb_frames)
        t1 = time.time()
        #video, audio, info = io.read_video(in_video_path, pts_unit='sec', start_pts=0, end_pts=10, output_format='TCHW')
        video = smart_resize(video, 256)
        t2 = time.time()
        torch.save(video, out_video_path)
        t3 = time.time()
        print(f"resize: {t2-t1}\nsave: {t3-t2}")
        #video = video.permute(0,2,3,1)
        #io.write_video(video_path, video, 15, video_codec='h264')

    
    for f in tqdm(train_files):
        in_video_path = os.path.join(dataset_dir, "train_dataset", f)
        out_video_path = os.path.join(resized_dir, "train_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from train")
    
    for f in tqdm(test_files):
        in_video_path = os.path.join(dataset_dir, "test_dataset", f)
        out_video_path = os.path.join(resized_dir, "test_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from test")
    for f in tqdm(experimental_files):
        in_video_path = os.path.join(dataset_dir, "experimental_dataset", f)
        out_video_path = os.path.join(resized_dir, "experimental_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from experimental")
    os.system(f"cp {os.path.join(dataset_dir, 'train_dataset', 'metadata.json')} {os.path.join(resized_dir, 'train_dataset', 'metadata.json')}")
    os.system(f"cp {os.path.join(dataset_dir, 'dataset.csv')} {os.path.join(resized_dir, 'dataset.csv')}")
    os.system(f"cp {os.path.join(dataset_dir, 'experimental_dataset', 'metadata.json')} {os.path.join(resized_dir, 'experimental_dataset', 'metadata.json')}")
    if errors:
        print(errors)
"""
use_small_dataset = False
if use_small_dataset:
    dataset_dir = resized_dir

nb_frames = 10

class VideoDataset(Dataset):
    """
    This Dataset takes a video and returns a tensor of shape [10, 3, 256, 256]
    That is 10 colored frames of 256x256 pixels.
    """
    def __init__(self, root_dir, dataset_choice="train", nb_frames=10):
        super().__init__()
        self.dataset_choice = dataset_choice
        if  self.dataset_choice == "train":
            self.root_dir = os.path.join(root_dir, "train_dataset")
        elif  self.dataset_choice == "test":
            self.root_dir = os.path.join(root_dir, "test_dataset")
        elif  self.dataset_choice == "experimental":
            self.root_dir = os.path.join(root_dir, "experimental_dataset")
        else:
            raise ValueError("choice must be 'train', 'test' or 'experimental'")

        with open(os.path.join(root_dir, "dataset.csv"), 'r') as file:
            reader = csv.reader(file)
            # read dataset.csv with id,label columns to create
            # a dict which associated label: id
            self.ids = {row[1][:-3] + "pt" : row[0] for row in reader}

        if self.dataset_choice == "test":
            self.data = None
        else:
            with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
                self.data= json.load(file)
                self.data = {k[:-3] + "pt" : (torch.tensor(float(1)) if v == 'fake' else torch.tensor(float(0))) for k, v in self.data.items()}

        #self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        #video, audio, info = io.read_video(video_path, pts_unit='sec')
        #video = extract_frames(video_path)
        video = torch.load(video_path)

        """
        video = video.permute(0,3,1,2)
        length = video.shape[0]
        video = video[[i*(length//(nb_frames)) for i in range(nb_frames)]]
        """
        # resize the data into a reglar shape of 256x256 and normalize it
        #video = smart_resize(video, 256) / 255
        video = video / 255

        ID = self.ids[self.video_files[idx]]
        if self.dataset_choice == "test":
            return video, ID
        else:
            label = self.data[self.video_files[idx]]
            return video, label, ID



train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)


# MODELE

class DeepfakeDetector(nn.Module):
    def __init__(self, nb_frames=10):
        super().__init__()
        self.dense = nn.Linear(nb_frames*3*256*256,1)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.flat(x)
        y = self.dense(y)
        y = self.sigmoid(y)
        return y

# LOGGING

"""
wandb.login(key="a446d513570a79c857317c3000584c5f6d6224f0")

run = wandb.init(
    project="automathon"
)
"""

train =  False

if train:
    # ENTRAINEMENT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    loss_fn = nn.MSELoss()
    model = DeepfakeDetector().to(device)
    print("Training model:")
    summary(model, input_size=(batch_size, 3, 10, 256, 256))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #loader = DataLoader(experimental_dataset, batch_size=2, shuffle=True)

    print("Training...")
    for epoch in range(epochs):
        for sample in tqdm(loader):
            optimizer.zero_grad()
            X, label, ID = sample
            X = X.to(device)
            label = label.to(device)
            label_pred = model(X)
            label = torch.unsqueeze(label,dim=1)
            loss = loss_fn(label, label_pred)
            loss.backward()
            optimizer.step()
            run.log({"loss": loss.item(), "epoch": epoch})

    ## TEST

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    ids = []
    labels = []
    print("Testing...")
    for sample in tqdm(loader):
        X, ID = sample
        #ID = ID[0]
        X = X.to(device)
        label_pred = model(X)
        ids.extend(list(ID))
        pred = (label_pred > 0.5).long()
        pred = pred.cpu().detach().numpy().tolist()
        labels.extend(pred)

    ### ENREGISTREMENT
    print("Saving...")
    tests = ["id,label\n"] + [f"{ID},{label_pred[0]}\n" for ID, label_pred in zip(ids, labels)]
    with open("submission.csv", "w") as file:
        file.writelines(tests)




def read_json():

    # Path to your metadata.json file
    json_file_path = os.path.join(dataset_dir, "train_dataset/metadata.json")
    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Count the occurrences of "fake" and "real" values
    fake_count = sum(1 for value in data.values() if value == "fake")
    real_count = sum(1 for value in data.values() if value == "real")

    print("fake:", fake_count)
    print("real:", real_count)


def rand_file_name(real_or_fake):

    # Path to your metadata.json file
    json_file_path = os.path.join(dataset_dir, "train_dataset/metadata.json")
    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    real_filenames = [filename for filename, label in data.items() if label == real_or_fake]
    if real_filenames:
        return random.choice(real_filenames)
    else:
        return None
    


def extract_frame(nb_frames):

    train_path = os.path.join(dataset_dir, "train_dataset")
    video_path = os.path.join(train_path, "xlewytmgee.mp4")

    reader = io.VideoReader(video_path)
    delta = 1

    # take 10 frames uniformly sampled from the video
    frames = []
    for i in range(nb_frames):
        reader.seek(delta)
        frame = next(reader)
        frames.append(frame['data'])

    return frames


def extract_frame3(video_path, output_dir,nb_frames, interval=1):
    container = av.open(video_path)
    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    saved_frames = 0  # Counter for the number of frames saved

    # Iterate over the video frames
    for frame in container.decode(video=0):
        # Check if the frame index is divisible by the interval
        if frame_index % interval == 0:
            # Convert the frame to a PIL Image
            image = frame.to_image()

            # Save the frame as an image file

            id_path = os.path.basename(video_path)
            id_path = os.path.splitext(id_path)[0]

            image.save(os.path.join(output_dir, f"frame_{id_path}_{frame_index}.jpg"))

            # Increment the counter for saved frames
            saved_frames += 1

            # Break out of the loop if 10 frames have been saved
            if saved_frames == nb_frames:
                break

        # Increment the frame index
        frame_index += 1



train_path = os.path.join(dataset_dir, "train_dataset")
video_path = os.path.join(train_path, "ryxykawftt.mp4") #fake


# ROOT directory for output images
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
root_directory = os.path.dirname(current_directory)

frames_dir = os.path.join(root_directory, "frames_folder")


extract_frames = True
if extract_frames:
    extract_frame3(video_path, frames_dir, nb_frames=10, interval = 10)

"""
def show_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out only the JPEG files
    jpeg_files = [file for file in files if file.endswith(".jpg") or file.endswith(".jpeg")]

    # Loop through each JPEG file and display it
    for jpeg_file in jpeg_files:
        file_path = os.path.join(folder_path, jpeg_file)
        image = Image.open(file_path)
        image.show()
"""



frames_dir = os.path.join(root_directory, "frames_folder")
frame_path = os.path.join(frames_dir, "frame_10.jpg")

import os
from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt

def detect_face(file_path):

    mtcnn = MTCNN(select_largest=False, device='cpu')

    image = Image.open(file_path).convert("RGB")

    # Detect faces in the image
    boxes, _ = mtcnn.detect(image)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Extract and save the face
            face_image = image.crop(box)
            face_image.save(os.path.join(os.getcwd(), f"face_{0}.jpg"))


def return_fake_real_paths(nb_videos):

    train_dir = "/raid/datasets/hackathon2024/train_dataset"


    # Path to your metadata.json file
    json_file_path = os.path.join(dataset_dir, "train_dataset/metadata.json")
    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    fake_paths = [file for file, label in data.items() if label == "fake"]
    real_paths = [file for file, label in data.items() if label == "real"]

    x = int(0.84 * nb_videos)
    y = int(0.16 * nb_videos)

    selected_fake_paths = []
    selected_real_paths = []

    while len(selected_fake_paths) < min(x, len(fake_paths)):
        sampled_path = random.choice(fake_paths)
        path = os.path.join(train_dir, sampled_path)
        if os.path.exists(path):
            selected_fake_paths.append(sampled_path)

    # Sample real paths
    while len(selected_real_paths) < min(y, len(real_paths)):
        sampled_path = random.choice(real_paths)
        path = os.path.join(train_dir, sampled_path)
        if os.path.exists(path):
            selected_real_paths.append(sampled_path)

    return selected_fake_paths, selected_real_paths
    

def make_light_dataset(nb_videos, light_fake_dir, light_real_dir):


    train_dir = "/raid/datasets/hackathon2024/train_dataset"
    fake_paths, real_paths = return_fake_real_paths(nb_videos)
    #fake videos 
    print("fake_paths: ", fake_paths)
    print("real_paths: ",real_paths)

    for fake_path in tqdm(fake_paths):
        path = os.path.join(train_dir, fake_path)
        extract_frame3(path, light_fake_dir, 10, interval = 10)

    for real_path in tqdm(real_paths):
        path = os.path.join(train_dir, real_path)
        extract_frame3(path, light_real_dir, 10, interval = 10)



def make_test_dataset():

    test_dir = "/raid/datasets/hackathon2024/test_dataset"

    file_paths = [file_path for file_path in os.listdir(test_dir)]
    output_dir = "test_folder"

    for file_path in file_paths:
        extract_frame3(file_path, output_dir, 10, interval = 10)

make_test_dataset()



def convert_frames_to_face():

    light_fake_dir = os.path.join(root_directory, "light_fake")
    light_real_dir = os.path.join(root_directory, "light_real")

    light_fake_files = os.listdir(light_fake_dir)
    light_real_files = os.listdir(light_real_dir)


    for real_path in tqdm(light_real_files):
        full_path = os.path.join(light_real_dir, real_path)
        detect_face(full_path)


    for fake_path in tqdm(light_fake_files):
        full_path = os.path.join(light_fake_dir, fake_path)
        detect_face(full_path)
  


light_fake_dir = os.path.join(root_directory, "light_fake")
light_real_dir = os.path.join(root_directory, "light_real")

heavy_fake_dir = os.path.join(root_directory, "heavy_fake")
heavy_real_dir = os.path.join(root_directory, "heavy_real")

#make_light_dataset(2000, heavy_fake_dir, heavy_real_dir)


def detect_face(file_path):

    mtcnn = MTCNN(select_largest=False, device='cpu')

    image = Image.open(file_path).convert("RGB")

    # Detect faces in the image
    boxes, _ = mtcnn.detect(image)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Extract and save the face
            face_image = image.crop(box)
            face_image.save(file_path)


    
#convert_frames_to_face()





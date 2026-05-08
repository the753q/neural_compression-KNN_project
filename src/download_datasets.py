# This script is used to create the datasets folder.
# The datasets are downloaded with kagglehub into some .cache directory,
# then copied into the datasets folder.
# The datasets folder is cleared when the script is run,
# however the kagglehub keeps track of the downloaded data.
# So, when the script is re-run the data is just copied freshly, not downloaded again.

import kagglehub
import shutil
import os
from glob import glob
import random
import subprocess
from PIL import Image
from pathlib import Path

DATASETS_DIR = "datasets"

def get_dataset(kaggle_path, name):
    print(f"Downloading dataset {name}")
    path = kagglehub.dataset_download(kaggle_path)
    print(f"Downloaded dataset {name} to: {path}")
    shutil.copytree(path, f"{DATASETS_DIR}/{name}", dirs_exist_ok=True)
    print(f"Copied dataset {name} to: {DATASETS_DIR}/{name}")

def get_df2k(convert_to_jpeg = False):
    print(f"Downloading dataset div2k")
    div2k_path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
    print(f"Downloading dataset flickr2k")
    flickr2k_path = kagglehub.dataset_download("daehoyang/flickr2k")

    DF2K_TRAIN_DIR = f"{DATASETS_DIR}/DF2K/train/train"
    DF2K_TEST_DIR = f"{DATASETS_DIR}/DF2K/test/test"

    os.makedirs(DF2K_TRAIN_DIR, exist_ok=True)
    os.makedirs(DF2K_TEST_DIR, exist_ok=True)

    sources = [
        os.path.join(div2k_path, "DIV2K_train_HR/DIV2K_train_HR/*.png"),
        os.path.join(div2k_path, "DIV2K_valid_HR/DIV2K_valid_HR/*.png"),
        os.path.join(flickr2k_path, "Flickr2K/*.png"),
    ]

    print(f"Convert to jpeg: {convert_to_jpeg}")

    # copy files into train
    print("Combining files: ")
    if convert_to_jpeg:
        for pattern in sources:
            for src_file in glob(pattern):
                dst_file = os.path.join(DF2K_TRAIN_DIR, Path(src_file).stem + ".jpg")
                Image.open(src_file).convert("RGB").save(dst_file, "JPEG", quality=95)
    else:
        for pattern in sources:
            subprocess.run(f"cp {pattern} {DF2K_TRAIN_DIR}/", shell=True)


    # separate test files
    all_files = os.listdir(DF2K_TRAIN_DIR)
    random.shuffle(all_files)
    test_files = all_files[:100]

    for f in test_files:
        shutil.move( os.path.join(DF2K_TRAIN_DIR, f),
                     os.path.join(DF2K_TEST_DIR, f))

    print(f"Train: {len(os.listdir(DF2K_TRAIN_DIR))} | Test: {len(os.listdir(DF2K_TEST_DIR))}")

def get_minecraft_screenshots():
    get_dataset(kaggle_path="sqdartemy/minecraft-screenshots-dataset-with-features", name="minecraft_screenshots")    
    
    MINECRAFT_TRAIN_DIR = f"{DATASETS_DIR}/minecraft_screenshots/screenshots/train"
    MINECRAFT_TEST_DIR = f"{DATASETS_DIR}/minecraft_screenshots/screenshots/test"

    print("Train test split.")
    os.rename(f"{DATASETS_DIR}/minecraft_screenshots/screenshots/screenshots",
             MINECRAFT_TRAIN_DIR)

    os.makedirs(MINECRAFT_TEST_DIR, exist_ok=True)
    all_files = os.listdir(MINECRAFT_TRAIN_DIR)
    random.shuffle(all_files)
    test_files = all_files[:100]

    for f in test_files:
        shutil.move( os.path.join(MINECRAFT_TRAIN_DIR, f),
                     os.path.join(MINECRAFT_TEST_DIR, f))

    print(f"Train: {len(os.listdir(MINECRAFT_TRAIN_DIR))} | Test: {len(os.listdir(MINECRAFT_TEST_DIR))}")

#remove old dataset dir, and create a new one
shutil.rmtree(DATASETS_DIR, ignore_errors=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

get_dataset(kaggle_path="priyerana/imagenet-10k", name="imagenet_10K")
get_minecraft_screenshots()
get_df2k(convert_to_jpeg=True)
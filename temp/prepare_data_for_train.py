import os, sys
os.chdir("/content/yolov10") # os.chdir("/Users/alex/Documents/Code/yolov10")
sys.path.append('.')
os.environ['HF_HOME'] = "/content/gdrive/MyDrive/cache/huggingface/datasets"

from ultralytics import settings

# Update the datasets directory setting
settings.update({"datasets_dir": "/content/gdrive/MyDrive/datasets"})

from ultralytics import YOLO  #  Reads settings
from datetime import datetime

from datasets import load_dataset, DownloadMode
import aiohttp
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

categories = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

# Load the DocLayNet dataset

download_url = ""  # Example: "https://ultralytics.com/assets/coco8.zip"
dataset = load_dataset(
    "ds4sd/DocLayNet",
    download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS,
    storage_options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total = 3600)}},
    trust_remote_code = True
)
print(dataset)

# Create the folders to hold the data in YOLO training format

dataset_root_dir = "/content/gdrive/MyDrive/datasets/try1"
# dataset_root_dir = "temp/data/try1"
dataset_folders = {
    "images" : {
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test"
    },
    "labels" : {
        "train" : "labels/train",
        "val"   : "labels/val",
        "test"  : "labels/test"
    }
}
for k1, v1 in dataset_folders.items():
    for k2, folder in v1.items():
        path = os.path.join(dataset_root_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)

# Create the dataset config YAML file

config_yaml_file = "doclaynet.yaml"
config_yaml_string = (
    "# YOLO training exercise, dataset config YAML\n" +
    "# Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n" +
    "# Prepared from the DocLayNet dataset, load_dataset(\"ds4sd/DocLayNet\")\n" +
    "#\n" +
    "# Train/val/test sets as\n" +
    "#     1) dir: path/to/imgs,\n" +
    "#     2) file: path/to/imgs.txt, or\n" +
    "#     3) list: [path/to/imgs1, path/to/imgs2, ..]\n" +
    "#\n" +
    f"path: {dataset_root_dir} # dataset root dir\n" +
    f"train: {dataset_folders['images']['train']} # train images (relative to 'path')\n"
    f"val: {dataset_folders['images']['val']} # val images (relative to 'path')\n" +
    f"test: {dataset_folders['images']['test']} # test images (optional)\n" +
    "\n" +
    "# Classes\n" +
    "names:\n" +
    "\n".join([f"  {i}: {name}" for i, name in enumerate(categories)]) +
    "\n\n" +
    "# Download script/URL (optional)\n" +
    f"download: {download_url}\n"
)
with open(os.path.join(dataset_root_dir, config_yaml_file), "w") as config_yaml:
    config_yaml.write(config_yaml_string)

# Generate the image and label files for each datapoint

dataset_split_names = {
    "train" : "train",
    "val"   : "validation",
    "test"  : "test"
}
dataset_split_sizes = {
    "train" : 10, # 69375,
    "val"   : 10, # 6489,
    "test"  : 10  # 4999
}
for split, s_name in dataset_split_names.items():
    # for d_index, datum in tqdm(enumerate(dataset[split_name]), total = len(dataset[split_name])):
    for d_index in tqdm(range(dataset_split_sizes[split])):
        datum = dataset[s_name][d_index]
        image_file = os.path.join(
            dataset_root_dir,
            dataset_folders["images"][split],
            f"{datum['image_id']:09}.jpg"
        )
        label_file = os.path.join(
            dataset_root_dir,
            dataset_folders["labels"][split],
            f"{datum['image_id']:09}.txt"
        )
        datum['image'].save(image_file, format = "jpeg")
        labels = []
        for obj in datum["objects"]:
            annot_id = obj["precedence"]  # Annotation (set-of-entities) ID
            if annot_id == 0:
                x_center = (obj['bbox'][0] + obj['bbox'][2] / 2) / datum['width']
                y_center = (obj['bbox'][1] + obj['bbox'][3] / 2) / datum['height']
                width    =  obj['bbox'][2] / datum['width']
                height   =  obj['bbox'][3] / datum['height']
                labels.append(f"{obj['category_id']} {x_center} {y_center} {width} {height}")
        if labels:
            with open(label_file, "w") as f:
                f.write("\n".join(labels))
            

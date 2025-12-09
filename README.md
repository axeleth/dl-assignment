# Deep Learning 2025 Dataset

## Overview

The **Deep_Learning_2025_Dataset** is a dataset designed for deep learning finel project involving microrobot pose and depth estimation.

## Directory Structure
The dataset is organized into subfolders located under the root directory:

```
/Deep_Learning_2025_Dataset/
├── P10_R0/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── …
│   └── P10_R0-depth.txt
├── P15_R15/
│   ├── image_001.jpg
│   ├── …
│   └── P15_R15-depth.txt
└── …
```

Each subfolder corresponds to a specific **Robot Pose**, defined by two angles:

- `P{pitch}_R{roll}`  
  - `Pitch`: The vertical tilt angle in degrees  
  - `Roll`: The horizontal rotation angle in degrees  

For example:
- `P10_R0/` means pitch = 10°, roll = 0°
- `P15_R15/` means pitch = 15°, roll = 15°

## Contents of Each Pose Folder

Each folder contains:
- Multiple **RGB images** captured at the specified pose (e.g., `image_001.jpg`, `image_002.jpg`, etc.)
- A **depth annotation text file** named `{pose}-depth.txt`, e.g., `P10_R0-depth.txt`

### Depth File Format

The depth text file contains the depth values corresponding to the images in that folder. Each line in the file represents the depth value for the corresponding image in the same order.

Example (`P10_R0-depth.txt`):
('video_14_stable_bot_l6s3_P10_R0_00255.jpg', 0.893)
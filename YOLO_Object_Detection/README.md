# YOLO Object Detection
Creating a custom YOLO object detection model **requires a specific folder structure** to organize your data and configuration files. This structure ensures that the training script can easily find and use your images, labels, and class information.

## Folder Structure
The most common and recommended folder structure for YOLO (specifically YOLOv5, which is widely used) looks like this:

```bash
    /Project 
    ├── /data
    │   └── data.yaml
    ├── /images
    │   ├── /train
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── /val
    │       ├── img3.jpg
    │       ├── img4.jpg
    │       └── ...
    ├── /labels
    │   ├── /train
    │   │   ├── img1.txt
    │   │   ├── img2.txt
    │   │   └── ...
    │   └── /val
    │       ├── img3.txt
    │       ├── img4.txt
    │       └── ...
    ├── requirments.txt
    ├── README.txt
    ├── license.txt
    ├── image_detect.py
    ├── video_detct.py
    └──train.py

```
(yolov5 repository files like train.py, detect.py, etc.)


- **images/:** This directory holds all your training and validation images.
- **labels/:** This directory contains the corresponding text files for your images. Each .txt file has the same name as its image counterpart.
- **data/data.yaml:** This is a configuration file that tells the model where to find the data and what the class names are.

## Files and Content
### 1. The images/ folder
- **images/train/:** Place all images you will use for training here. This is typically the largest portion of your dataset (e.g., 80%).

- **images/val/:** Place all images you will use for validation here. This set is used during training to monitor performance and prevent overfitting (e.g., 20%). 
- **A third folder, /test,** is optional and used for final model evaluation after training.

### 2. The labels/ folder
Each image in the images folder must have a corresponding .txt file with the exact same name in the labels folder.

#### For your 3 objects, the contents of each .txt file will follow a specific format:
```bash
<class_id> <x_center> <y_center> <width> <height>
```
- **<class_id>:** An integer representing your object class, starting from 0. Since you have 3 objects, your classes will be 0, 1, and 2. For example, 0 for "book", 1 for "pen", and 2 for "notebook". The order of these IDs must match the order in your custom_data.yaml file.

- **<x_center> <y_center> <.width> <.height>:** These are the bounding box coordinates, normalized to a value between 0.0 and 1.0. This means the values are a fraction of the total image width and height.

#### Example:
If you have an image img1.jpg with a resolution of 640x480, and a "cat" is located at the top-left corner with coordinates (50, 50) and a width of 100 and height of 200, its label file img1.txt would contain:
```bash
    class_id: 0

    x_center: (50+100/2)/640=100/640
    approx0.156

    y_center: (50+200/2)/480=150/480
    approx0.312

    width: 100/640
    approx0.156

    height: 200/480
    approx0.417

    The contents of img1.txt would then be:
    0 0.156 0.312 0.156 0.417
```
### 3. The data.yaml file
This YAML file is crucial for configuring your training. You must create it and place it in the data/ folder. It should contain the following information:

```bash
# YAML File content

train: ../images/train
val: ../images/val

nc: 2

names: ['cat', 'dog']
```
- **train:** Specifies the path to your training images.

- **val:** Specifies the path to your validation images.

- **nc:** The number of classes, which in your case is 2.

- **names:** A list of the class names in the exact same order as their numerical IDs (0, 1, 2).
---
### After Training :
After Traning you will get folder get **new folder** with name **runs** where you will **contain** all the **weights, loss etc.**

---

## How to Use YOLO for Object Detection
**1. Setup Your Environment**

First, set up a clean workspace. Create a virtual environment to manage dependencies and activate it.

```Bash
python -m venv yolovenv
source yolovenv/bin/activate  # On Windows, use `yolovenv\Scripts\activate`
```
**2. Install Dependencies**

Install all the necessary libraries from a requirements.txt file. This ensures you have all the required packages, like PyTorch and OpenCV, for YOLO to run.

```Bash
pip install -r requirements.txt
```
**3. Prepare Your Dataset**

Organize your images and their corresponding labels into the correct folder structure.

Place all images in project/images/train and project/images/val.

Place the corresponding label files in project/labels/train and project/labels/val.

Ensure all your images are properly annotated. Each image must have a .txt file with the same name containing the normalized bounding box coordinates and class IDs. If you need to annotate your images, you can use a tool like Roboflow or LabelImg.

**4. Train Your Model**

With your dataset ready, run the training script. This command starts the training process using a pre-trained model and your custom data.

```Bash
python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt
```
**--img 640:** Specifies the image size for training.

**--batch 16:** Sets the number of images per batch.

**--epochs 100:** Defines the number of training cycles.

**--data custom_data.yaml:** Points to your dataset configuration file.

**--weights yolov5s.pt:** Uses a pre-trained YOLOv5 small model as a starting point for faster and more accurate training.

**5. Use Your Trained Model for Inference**

After training is complete, your final trained model will be saved as best.pt inside the newly created runs/train/exp directory. You can use this model to detect objects on new images or videos.

- **Detect objects on an image:**

    ```Bash
    python detect.py --weights runs/train/exp/weights/best.pt --source your_image.jpg
    ```
- **Detect objects on a video:**

    ```Bash
    python detect.py --weights runs/train/exp/weights/best.pt --source your_video.mp4
    ```
- **Detect objects using a webcam:**

    ```Bash
    python detect.py --weights runs/train/exp/weights/best.pt --source 0
    ```

---
### Other Resources :
- **train.py** can be used to train the small model but make sure your folder structure follow yolo requirements. 
- **image_detect.py** can be used to create the bounding boxes with trained model on images.
- **video_detect.py** can be used to create teh bounding boses with tranied model on videos. 
- This Repo is exact small structure for yolo, You can clone and run after necessary libaray installation, Then you can see the power of YOLO.

---
- **License** : MIT
- **Contact Information** : shergillkuldeep@outlook.com


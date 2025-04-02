import os
import cv2
import warnings
import torch
import numpy as np
from torchvision import models, transforms
from fastai.vision.all import load_learner, PILImage
from picamera2 import Picamera2
from PIL import Image

# Load all classes
with open("model_weight.txt", "r") as file:
    IMAGENET_CLASSES = {i: line.strip() for i, line in enumerate(file.readlines())}

# First Layer: Non-garbage object classes
NON_GARBAGE_CLASSES = [
    "sports ball", "teddy bear", "laptop", "cell phone",
    "remote control", "keyboard", "mouse", "chair", "couch",
    "potted plant", "clock", "hard disc", "rubber eraser", 
    "flute", "drumstick", "pole", "screwdriver", "studio couch", 
    "coil", "dining table", "rifle", "sliding door", "revolver",
    "mixing bowl"
]

# Thresholds
CONFIDENCE_THRESHOLD = 0.3
MAX_RETRIES = 5

def load_pth_model(pth_model_path):
    model = models.efficientnet_b5(pretrained=False)
    model.load_state_dict(torch.load(pth_model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def capture_image(picam2):
    print("Adjusting focus... Please hold the object steady.")
    picam2.set_controls({"AfMode": 1})
    picam2.autofocus_cycle()
    
    print("Locked! Capturing image...")
    i = 1
    while os.path.exists(f"test{i}.jpg"):
        i += 1
    image_path = f"test{i}.jpg"
    
    picam2.capture_file(image_path)
    print(f"Image captured: {image_path}")
    return image_path
def classify_with_pth(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_class = probabilities.max(0)

    class_name = IMAGENET_CLASSES[predicted_class.item()]
    
    print(f"First layer: {class_name} (Confidence: {confidence:.2f})")
    
    if confidence < CONFIDENCE_THRESHOLD:
        print("Low confidence. Retrying...")
        return None

    return class_name

def classify_with_pkl(model_path, image_path):
    try:
        learn = load_learner(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    img = PILImage.create(image_path)
    pred, _, probs = learn.predict(img)

    print(f"Garbage Model Final Decision: {pred} (Confidence: {probs.max():.2f})")
    return pred

def is_non_garbage_item(item):
    return item.lower() in NON_GARBAGE_CLASSES
def classify_with_pth(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_class = probabilities.max(0)

    class_name = IMAGENET_CLASSES[predicted_class.item()]
    
    print(f"First layer: {class_name} (Confidence: {confidence:.2f})")
    
    if confidence < CONFIDENCE_THRESHOLD:
        print("Low confidence. Retrying...")
        return None

    return class_name

def classify_with_pkl(model_path, image_path):
    try:
        learn = load_learner(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    img = PILImage.create(image_path)
    pred, _, probs = learn.predict(img)

    print(f"Garbage Model Final Decision: {pred} (Confidence: {probs.max():.2f})")
    return pred

def is_non_garbage_item(item):
    return item.lower() in NON_GARBAGE_CLASSES

def run_ml_pipeline():
    warnings.filterwarnings("ignore")
    pth_model_path = "first_layer.pth"
    pkl_model_path = "garbage_model.pkl"

    pth_model = load_pth_model(pth_model_path)

    try:
        picam2 = Picamera2(0)
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()

        print("Detection started...")
        retry_count = 0

        while True:
            image_path = capture_image(picam2)
            detected_object = classify_with_pth(pth_model, image_path)

            if detected_object is None:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print("Stopping after max retries.")
                    break
                continue

            if is_non_garbage_item(detected_object):
                print(f"Detected non-garbage object: {detected_object}. Stopping.")
                break

            print(f"Passing {detected_object} to the garbage classifier...")
            classify_with_pkl(pkl_model_path, image_path)
            break

    finally:
        print("Cleaning up camera...")
        try:
            picam2.stop()
            picam2.close()
        except Exception as e:
            print(f"Camera cleanup error: {e}")

        print("ML pipeline finished.")

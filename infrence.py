from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as T
import os

# Load the pre-trained model
model = YOLO("./weights/best.pt")

# Load and preprocess the image
img_path = "./testing_images/sample_8.jpg"
img = Image.open(img_path).convert('RGB')

# Resize the image to the closest multiple of 32
def resize_image(image, stride=32):
    width, height = image.size
    new_width = (width + stride - 1) // stride * stride
    new_height = (height + stride - 1) // stride * stride
    # Resize while maintaining the aspect ratio
    resize_transform = T.Resize((new_height, new_width))
    image_resized = resize_transform(image)
    return image_resized

img_resized = resize_image(img)

# Convert the image to a NumPy array and normalize to 0-1
img_np = np.array(img_resized) / 255.0

# Change data layout from HWC to CHW
img_np = np.transpose(img_np, (2, 0, 1))

# Convert to a tensor and add batch dimension
img_tensor = torch.from_numpy(img_np).unsqueeze(0)

# Perform detection using the model
results = model(img_tensor)

# Ensure results is not a list
if isinstance(results, list):
    results = results[0]

if hasattr(results, 'boxes') and results.boxes:
    # Move boxes to CPU and extract class labels
    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes as numpy array
    class_labels = results.boxes.cls.cpu().numpy()  # Class labels as numpy array
    fruit_counts = {}

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(img_resized)
    font = ImageFont.load_default()  # You can use a more suitable font if available

    for i, class_id in enumerate(class_labels):
        class_name = results.names[int(class_id)]
        if class_name in fruit_counts:
            fruit_counts[class_name] += 1
        else:
            fruit_counts[class_name] = 1

        box = boxes[i]
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=2)
        draw.text((box[0], box[1] - 10), f"{class_name}: {fruit_counts[class_name]}", fill="red", font=font)

    # Ensure the output directory exists
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output file name
    base_filename = os.path.basename(img_path)
    filename, ext = os.path.splitext(base_filename)
    output_filename = f"{filename}_detected.png"
    output_path = os.path.join(output_dir, output_filename)

    # Save or display the annotated image
    img_resized.save(output_path)
    print(f"Annotated image saved at {output_path}")
else:
    print("No detections were made.")

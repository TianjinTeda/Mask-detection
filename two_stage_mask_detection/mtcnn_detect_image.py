from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from inspect_images import class_names
from torchvision import transforms
from copy import deepcopy

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create face detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=True, post_process=False, device='cuda:0')

# Load the mask detector model
model = torch.load('model/finetuned_model_resnet18.pth')

# Read the image
image = cv2.imread('data/test_image/test_image_10.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_for_draw = deepcopy(image)
image = Image.fromarray(image)

# Detect the faces in the image and detect the mask
faces = mtcnn(image)

# Get the bounding boxes of faces
boxes, _, _ = mtcnn.detect(image, landmarks=True)

# Set the font for writing label
font = cv2.FONT_HERSHEY_SIMPLEX

# Detect mask and draw bounding boxes
for face, box in zip(faces, boxes):
    # Normalize the testing image
    face_image = deepcopy(face)
    face_image /= 255.0
    face_image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(face_image)

    # Detect the masks
    face_image = face_image.to(device).unsqueeze(0)
    output = model(face_image)
    _, pred = torch.max(output, 1)

    # Visualization
    if class_names[pred] == 'mask':
        cv2.rectangle(image_for_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        cv2.putText(image_for_draw, class_names[pred.item()], (int(box[0]), int(box[1])), font, 0.5, (0, 255, 0), 1)
    else:
        cv2.rectangle(image_for_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
        cv2.putText(image_for_draw, class_names[pred.item()], (int(box[0]), int(box[1])), font, 0.5, (255, 0, 0), 1)

image_for_draw = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
cv2.imshow('Mask detection', image_for_draw)
cv2.waitKey(0)
cv2.imwrite('data/test_image/result_image_10_res18.jpg', image_for_draw)
import os
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from inspect_images import class_names
from torchvision import transforms
from copy import deepcopy
from utils import iou
import shutil

# Set parameters
iou_threshold = 0.0

# Initialize statistics variables
num_faces = 0
num_fake_faces = 0
num_positive = 0
num_negative = 0
num_true_positive = 0
num_false_positive = 0
num_true_negative = 0
num_false_negative = 0

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create face detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=True, post_process=False, device='cuda:0')

# Load the mask detector model
model = torch.load('model/finetuned_model_resnet18_nofake.pth')

# Dataset path
path = 'data/test_dataset/'

num_image = 2865
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(0, num_image):
    flag = False

    # Read the image
    image = cv2.imread(path + str(i) + '.jpg')
    size = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_draw = deepcopy(image)
    image = Image.fromarray(image)

    # Get the ground truth bounding boxes and labels
    bounding_boxes = []
    with open(path + str(i) + '.txt', 'r') as f:
        content = f.readlines()
    if len(content) == 0:
        continue
    else:
        for c in content:
            if len([float(x) for x in c[:-1].split(' ')]) != 5:
                flag = True
                break
            label, x, y, w, h = [float(x) for x in c[:-1].split(' ')]
            x = size[1] * x
            y = size[0] * y
            w = size[1] * w
            h = size[0] * h
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            bounding_boxes.append((x1, y1, x2, y2, label))

            # Update the statistics information
            num_faces += 1
            if label == 0:
                num_negative += 1
            else:
                num_positive += 1

    if flag:
        continue

    # Detect the faces in the image and detect the mask
    faces = mtcnn(image)

    # Get the bounding boxes of faces
    boxes, _, _ = mtcnn.detect(image, landmarks=True)

    if faces == None:

        continue

    # Detect mask and draw bounding boxes
    count = 0
    for face, box in zip(faces, boxes):
        # Normalize the testing image
        face_image = deepcopy(face)
        face_image /= 255.0
        face_image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(face_image)

        # Detect the masks
        face_image = face_image.to(device).unsqueeze(0)
        output = model(face_image)
        _, pred = torch.max(output, 1)

        # Check which ground truth box is closed to the predicted box
        max_iou = -1
        closest_box = None
        for gt_box in bounding_boxes:
            iou_value = iou(box, gt_box)
            if iou_value > max_iou:
                max_iou = iou_value
                closest_box = gt_box

        # Update the statistics information
        if max_iou <= iou_threshold:
            num_fake_faces += 1
            continue
        else:
            prediction = 1 - pred.item()
            if prediction == 0 and closest_box[4] == 0:
                num_true_negative += 1
            elif prediction == 0 and closest_box[4] == 1:
                num_false_negative += 1
            elif prediction == 1 and closest_box[4] == 0:
                num_false_positive += 1
            elif prediction == 1 and closest_box[4] == 1:
                num_true_positive += 1
            count += 1

    #if count == len(bounding_boxes):
        #print(i)
    #    shutil.copyfile(path + str(i) + '.jpg', 'C:\\Users\\13753\\Desktop\\test_dataset\\' + str(i) + '.jpg')
    #    shutil.copyfile(path + str(i) + '.txt', 'C:\\Users\\13753\\Desktop\\test_dataset\\' + str(i) + '.txt')

    print(i)

print('Statistics information: ')
print('Number of faces in the testing dataset: ' + str(num_faces))
print('Number of fake faces that the model detect: ' + str(num_fake_faces))
print('Number of faces with a mask in the testing dataset: ' + str(num_positive))
print('Number of faces without a mask in the testing dataset: ' + str(num_negative))
print('Number of true positive: ' + str(num_true_positive))
print('Number of false positive: ' + str(num_false_positive))
print('Number of true negative: ' + str(num_true_negative))
print('Number of false negative: ' + str(num_false_negative))

print('Recall: ' + str(num_true_positive / (num_true_positive + num_false_negative)))
print('Precision: ' + str(num_true_positive / (num_true_positive + num_false_positive)))
print('Accuracy: ' + str((num_true_positive + num_true_negative) / (num_true_positive + num_true_negative + num_false_positive + num_false_negative)))

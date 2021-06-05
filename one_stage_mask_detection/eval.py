from yolo import YOLO
import cv2
from PIL import Image
import torch
from torchvision import transforms
from copy import deepcopy
import shutil

def iou(box_1, box_2):
    box1_x1 = box_1[0]
    box1_y1 = box_1[1]
    box1_x2 = box_1[2]
    box1_y2 = box_1[3]

    box2_x1 = box_2[0]
    box2_y1 = box_2[1]
    box2_x2 = box_2[2]
    box2_y2 = box_2[3]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6) # Add a small number for smoothing

# Load the yolo model
yolo = YOLO()

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

# Dataset path
path = 'data/test_dataset/'

num_image = 2865
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(0, num_image):
    flag = False

    # Read the image
    image = Image.open(path + str(i) + '.jpg')
    size = image.size

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
            x = size[0] * x
            y = size[1] * y
            w = size[0] * w
            h = size[1] * h
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

    label, boxes = yolo.detect_image(image, mode='eval')
    if len(label) == 0:
        continue

    # Detect mask and draw bounding boxes
    count = 0
    for label, box in zip(label, boxes):

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
            prediction = 1 - label
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
    #    print(i)
    #    shutil.copyfile(path + str(i) + '.jpg', 'C:\\Users\\13753\\Desktop\\test_dataset\\test' + str(i) + '.jpg')
    #    shutil.copyfile(path + str(i) + '.txt', 'C:\\Users\\13753\\Desktop\\test_dataset\\test' + str(i) + '.txt')
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

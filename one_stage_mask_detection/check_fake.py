from yolo import YOLO
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch

image = Image.open('data/new_mask_data/105.jpg')
size = image.size
bounding_boxes = []
print(size)

with open('data/new_mask_data/105.txt', 'r') as f:
    content = f.readlines()
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
        bounding_boxes.append([x1, y1, x2, y2])
'''
draw = ImageDraw.Draw(image)
for box in bounding_boxes:
    draw.rectangle(
        box,
        outline=(0,0,255))
image.save('result.jpg')
'''
image = cv2.imread('data/new_mask_data/105.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image_for_draw = deepcopy(image)
#image = Image.fromarray(image)
for box in bounding_boxes:
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite('result_105.jpg', image)
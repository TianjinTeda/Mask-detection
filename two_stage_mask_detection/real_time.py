from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from facenet_pytorch import MTCNN
import torch
from inspect_images import class_names
from torchvision import transforms
from copy import deepcopy
from PIL import Image

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create face detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=True, post_process=False, device='cuda:0')

# Load the mask detector model
model = torch.load('model/finetuned_model_resnet18.pth')

#cv2.dnn.readNetFromTorch()
# Set the font for writing label
font = cv2.FONT_HERSHEY_SIMPLEX

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


while True:
    frame = vs.read()
    #frame = imutils.resize(frame, width=400)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_for_draw = deepcopy(image)
    image = Image.fromarray(image)

    faces = mtcnn(image)
    boxes, _, _ = mtcnn.detect(image, landmarks=True)

    # When there is no face
    if faces == None:
        image_for_draw = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
        continue

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

    # Write the new video
    image_for_draw = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", image_for_draw)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
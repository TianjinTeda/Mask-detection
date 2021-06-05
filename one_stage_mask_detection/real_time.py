from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
from copy import deepcopy
from PIL import Image
from yolo import YOLO

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

yolo = YOLO()

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
    size = image.size

    label, boxes = yolo.detect_image(image, mode='eval')
    if len(label) == 0:
        image_for_draw = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
        continue

    # Detect mask and draw bounding boxes
    for label, box in zip(label, boxes):

        # Visualization
        if label == 0:
            cv2.rectangle(image_for_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            cv2.putText(image_for_draw, 'mask', (int(box[0]), int(box[1])), font, 0.5, (0, 255, 0), 1)
        else:
            cv2.rectangle(image_for_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            cv2.putText(image_for_draw, 'no mask', (int(box[0]), int(box[1])), font, 0.5, (255, 0, 0), 1)

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
import cv2
import time
import numpy as np
from yolo import YOLO
from PIL import Image


def detect_image(image_path):
    print('Start detect!')
    yolo = YOLO()
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
        pass
    else:
        r_image = yolo.detect_image(image)
        r_image.save(image_path.split('.')[0] + '_result.png')
    print('Finish detect!')

detect_image('105.jpg')
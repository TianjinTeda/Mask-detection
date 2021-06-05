from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from inspect_images import class_names
from torchvision import transforms
from copy import deepcopy

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the mask detector model
model = torch.load('model/finetuned_model_resnet50.pth')

# Set the video input and output path
video_path = 'data/test_video/test_video_2.mp4'
output_path = 'data/test_video/video_result_2.mp4'

# Capture the video
vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")

# Get basic information of the video
video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Video information: ')
print('fps: ' + str(video_fps))
print('Video size: '+ str(video_size))

# Create face detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=True, post_process=False, device='cuda:0')

# Set the output
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

# Set the font for writing label
font = cv2.FONT_HERSHEY_SIMPLEX

frame_count = 0
# Detect the video
while True:
    return_value, frame = vid.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_for_draw = deepcopy(image)
    image = Image.fromarray(image)

    faces = mtcnn(image)
    boxes, _, _ = mtcnn.detect(image, landmarks=True)

    # When there is no face
    if faces == None:
        image_for_draw = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
        out.write(image_for_draw)
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
    out.write(image_for_draw)

    print(frame_count)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when it is finished
out.release()
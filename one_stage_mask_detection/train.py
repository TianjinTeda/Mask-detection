import os
import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from nets.yolo4 import YoloBody
from nets.yolo_training import YOLOLoss, Generator


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, optimizer,
                  lr_scheduler):
    total_loss = 0
    val_loss = 0
    print('\n' + '-' * 10 + 'Train one epoch.' + '-' * 10)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Start Training.')

    for iteration in range(epoch_size):
        net.train()
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss

        net.eval()
        val_losses = []
        images, targets = next(genval)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        with torch.set_grad_enabled(False):
            outputs = net(images)
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                val_losses.append(loss_item[0])
            cur_val_loss = sum(val_losses)
        val_loss += cur_val_loss

        waste_time = time.time() - start_time
        if iteration == 0 or (iteration + 1) % 10 == 0:
            print('step:' + str(iteration + 1) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (
            total_loss / (iteration + 1), waste_time))
    print('Finish Training.')
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    return total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)


#input_shape = (416,416)
input_shape = (608, 608)

Cosine_lr = True
mosaic = True
Cuda = True
smoooth_label = 0.03

train_annotation_path = 'model_data/mask_train.txt'
val_annotation_path = 'model_data/mask_val.txt'

anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/mask_classes.txt'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)

model = YoloBody(len(anchors[0]), num_classes)
model_path = "model_data/yolov4_coco_pretrained_weights.pth"
#model_path = "model_data/yolov4_maskdetect_weights0.pth"
print('Loading pretrained model weights.')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print('Finished!')

if Cuda:
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()

yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1,2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label, Cuda))
# read train lines and val lines
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

lr = 1e-3
Batch_size = 4
Init_Epoch = 0
Freeze_Epoch = 25

optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
if Cosine_lr:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
else:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

gen = Generator(Batch_size, train_lines, (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
gen_val = Generator(Batch_size, val_lines, (input_shape[0], input_shape[1])).generate(mosaic=False)

epoch_size = int(max(1, num_train // Batch_size // 2.5)) if mosaic else max(1, num_train // Batch_size)
epoch_size_val = num_val // Batch_size
for param in model.backbone.parameters():
    param.requires_grad = False

best_loss = 99999999.0
best_model_weights = copy.deepcopy(net.state_dict())
for epoch in range(Init_Epoch, Freeze_Epoch):
    total_loss, val_loss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                         Freeze_Epoch, Cuda, optimizer, lr_scheduler)
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
    with open('total_loss.csv', mode='a+') as total_loss_file:
        total_loss_file.write(str(total_loss.item()) + '\n')
torch.save(best_model_weights, 'model_data/yolov4_maskdetect_weights0.pth')

lr = 1e-4
Batch_size = 2
Freeze_Epoch = 25
Unfreeze_Epoch = 50

optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
if Cosine_lr:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
else:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

gen = Generator(Batch_size, train_lines, (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
gen_val = Generator(Batch_size, val_lines, (input_shape[0], input_shape[1])).generate(mosaic=False)

epoch_size = int(max(1, num_train // Batch_size // 2.5)) if mosaic else max(1, num_train // Batch_size)
epoch_size_val = num_val // Batch_size
for param in model.backbone.parameters():
    param.requires_grad = True

for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
    total_loss, val_loss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                         Unfreeze_Epoch, Cuda, optimizer, lr_scheduler)
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
    with open('total_loss.csv', mode='a+') as total_loss_file:
        total_loss_file.write(str(total_loss.item()) + '\n')
torch.save(best_model_weights, 'model_data/yolov4_maskdetect_weights1.pth')
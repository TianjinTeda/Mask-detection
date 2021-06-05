from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import copy
from inspect_images import dataset_sizes, mask_data_loader

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fine_tuning(model, criterion, optimizer, scheduler, num_epochs=30):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            loss = 0.0
            num_correct = 0
            for inputs, labels in mask_data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    current_loss = criterion(outputs, labels)
                    if phase == 'train':
                        current_loss.backward()
                        optimizer.step()
                loss += current_loss.item() * inputs.size(0)
                num_correct += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = loss / dataset_sizes[phase]
            epoch_acc = num_correct.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Store the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load the best model and return
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # Load the pre-trained classifier model
    resnet_18 = models.resnet18(pretrained=True)

    # Record the number of input features of the last fully-connected layer
    num_in_features = resnet_18.fc.in_features

    # Set the last layer's output to 2
    resnet_18.fc = nn.Linear(num_in_features, 2)
    resnet_18 = resnet_18.to(device)

    # Use the cross entropy loss
    bce_criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    adam_optimizer = optim.Adam(resnet_18.parameters())

    # Decay LR by a factor of 0.1 every 10 epochs
    lr_scheduler = lr_scheduler.StepLR(adam_optimizer, step_size=6, gamma=0.1)
    resnet_18 = fine_tuning(resnet_18, bce_criterion, adam_optimizer, lr_scheduler, num_epochs=30)
    torch.save(resnet_18, 'model/finetuned_model_resnet18_nofake.pth')

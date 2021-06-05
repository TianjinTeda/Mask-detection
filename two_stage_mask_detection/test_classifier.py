from __future__ import print_function, division

import torch
import matplotlib.pyplot as plt
from inspect_images import mask_data_loader, class_names
import numpy as np
import pylab

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def test_model(model, num_images=16):
    was_training = model.training
    model.eval()
    image_count = 0
    fig = plt.figure(figsize=(12, 12), dpi=80)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(mask_data_loader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(inputs)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(num_images):
                image_count += 1
                ax = plt.subplot(num_images // 4, 4, image_count)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                x = torch.zeros_like(inputs.cpu().data[j])
                for c in range(0, 3):
                    x[c] = inputs.data[j][c] * std[c] + mean[c]

                plt.imshow(np.transpose(x, (1, 2, 0)))

            pylab.show()
            if image_count == num_images:
                model.train(mode=was_training)
                return


if __name__ == '__main__':
    model = torch.load('model/finetuned_model_resnet18.pth')
    test_model(model)
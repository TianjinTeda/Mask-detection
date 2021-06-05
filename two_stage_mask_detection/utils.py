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
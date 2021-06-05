import os
import shutil

#目标文件夹，此处为相对路径，也可以改为绝对路径
# determination = 'target/'
determination = r'C:\Users\13753\Desktop\TUDelft\computer vision\mask_detection\data\self-built-masked-face-recognition-dataset\cele_mask'

#源文件夹路径
# path = r'E:\数据集\CUB_200_2011\CUB_200_2011\images'
path = 'data/mask/train/mask'
new_path = r'C:\Users\13753\Desktop\store\train\mask'
#path_2 = r'C:\Users\13753\Desktop\TUDelft\computer vision\mask_detection\data\self-built-masked-face-recognition-dataset\selected_image'
#folders= os.listdir(path)
# print(folders)


count = 0
for root, ds, fs in os.walk(path):
    for file in fs:
        name = file.split('.')
        print(name)
        if 'augmented' in name[0]:
            os.rename(os.path.join(path, file), os.path.join(new_path, file))
        #if name[1] == 'jpg':
            #os.rename(os.path.join(path, file), os.path.join(new_path, str(count) + '.jpg'))
            #os.rename(os.path.join(path, name[0] + '.txt'), os.path.join(new_path, str(count) + '.txt'))
            #count += 1
            #print(count)

'''
for folder in folders:
    dir = path + '\\' +  str(folder)
    files = os.listdir(dir)
    for file in files:
        os.rename(os.path.join(dir, file))
        count += 1
'''

'''
count = 0
count_2 = 0
mask = 0
no_mask = 0
for i in range(0, 2865):

    with open('data/test_dataset/' + str(i) + '.txt', 'r') as f:
        content = f.readlines()
        res = [x[:-1].split(' ') for x in content]
        for x in res:
            if x[0] == '0':
                no_mask += 1
            else:
                mask += 1

    #if no_mask == 0:
    #    os.rename('data/temporal/' + str(i) + '.jpg', 'data/test_dataset/' + 'tt'+str(i) + '.jpg')
    #    os.rename('data/temporal/' + str(i) + '.txt', 'data/test_Dataset/' + 'tt'+str(i) + '.txt')
    #print(i)
    #if no_mask == 0:
    #    count += mask
    #    print(str(i) + '\t' + str(count))
    print(str(i) + '\t' + 'No mask: ' + str(no_mask) + '\t' + 'Mask: ' + str(mask))
'''
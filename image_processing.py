import os
import cv2
import numpy as np

folder_path = 'C:/Users/user/Desktop/images/'
classes = ['angry','happy']
split_data = ['train','validation']
num_classes = len(classes)
print(num_classes)
X_train = []
Y_train = []
X_test = []
Y_test = []

img_train_data = []
for index, emotion in enumerate(classes):
    label = [0 for i in range(num_classes)]
    label[index] = 1
    print(label)
    for split in split_data:
        print(split)
        print(emotion)
        image_dir = folder_path + split + '/' + emotion + '/'
        if split == split_data[0]:
            for top, dir, f in os.walk(image_dir):
                for filename in f:
                    print(filename)
                    img = cv2.imread(image_dir + filename)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img_train_data.append(img)
                    X_train.append(img/255)
                    Y_train.append(label)
        else:
            for top, dir, f in os.walk(image_dir):
                for filename in f:
                    img = cv2.imread(image_dir + filename)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    print(filename)
                    X_test.append(img/255)
                    Y_test.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape)

np.save('dataset/X_train.npy',X_train)
np.save('dataset/X_test.npy',X_test)
np.save('dataset/Y_train.npy',Y_train)
np.save('dataset/Y_test.npy',Y_test)
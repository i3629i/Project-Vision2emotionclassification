import numpy as np
import os,h5py
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Dense,Activation,Flatten,MaxPooling2D,Dropout
from keras.models import Model,Sequential, load_model
from tensorflow.python.client import device_lib
from keras.utils import Sequence

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix

print(device_lib.list_local_devices())

classes = ['angry','happy']

x_train = np.load(os.path.join('dataset/X_train.npy'))
y_train = np.load(os.path.join('dataset/Y_train.npy'))

x_test = np.load(os.path.join('dataset/X_test.npy'))
y_test = np.load(os.path.join('dataset/Y_test.npy'))

print(x_train.shape)
print(x_test.shape)
#
x_train = x_train.reshape((-1,48,48,1))
x_test = x_test.reshape((-1,48,48,1))


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False, # 각 이미지의 평균을 0으로 만들겠다.
    samplewise_std_normalization=False, # 정규화 하는 작업
    brightness_range=[0.5,1.5], # 원래 밝기가 1.0   0.5에서 1.5까지 랜덤으로 설정
    zoom_range=[0.8,1.1], # 이미지 크기를 80퍼 줄이고 110퍼 까지 늘리는
    rotation_range=15., # -15도에서 15도 까지 돌린다
    channel_shift_range= 25,
    horizontal_flip= True
)
test_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)

# flow 는 batch_size만큼 생성할수 있게 만드는 메서드
train_batch_gen = train_datagen.flow(x_train, y_train,batch_size = 100, shuffle=True)
test_batch_gen = test_datagen.flow(x_test, y_test,shuffle=False)


model = Sequential()
model.add(Conv2D(24, kernel_size = 5, strides=1, padding='same', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,padding='same'))

model.add(Conv2D(48, kernel_size = 5, strides=1, padding='same', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,padding='same'))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))


model.add(Dense(len(classes)))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])

model.fit(x_train,y_train, batch_size= 48,epochs=100, validation_data=(x_test,y_test),
        callbacks=[
        ModelCheckpoint('models/4_emotions.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=3, verbose=1, mode='auto', min_lr=1e-05)
        ]
          )

model.summary()


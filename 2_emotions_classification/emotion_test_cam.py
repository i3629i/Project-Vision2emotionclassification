import cv2
import dlib
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('model/emotion2.h5')

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)


while True:

    ret,frame = cap.read()

    if not ret:
        exit


    faces = detector(frame)



    try:
        face = faces[0]
        x1 = face.left()
        x2 = face.right()
        y1 = face.bottom()
        y2 = face.top()
        frame = cv2.rectangle(frame, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255),thickness=2)
    except:
        cv2.imshow('img2', frame)

    test_image = frame[y2:y1,x1:x2]
    image = cv2.resize(test_image,dsize=(48,48))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    pred = model.predict(image.reshape(1,48,48,1))
    print(pred)

    if pred[0][0] >= 0.5:
        cv2.putText(frame,'Angry',(50,50), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,0))
    elif pred[0][1] >= 0.5:
        cv2.putText(frame,'Happy',(50,50), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,0))
    elif pred[0][2] >= 0.5:
        cv2.putText(frame,'sad',(50,50), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,0))
    else:
        cv2.putText(frame, 'neutral', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 255, 0))
    test_image = cv2.imshow('img1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# img = cv2.imread('test1.jpg',cv2.IMREAD_GRAYSCALE)
# face_img = detector(img,1)
#
#
# face = face_img[0]
# print(face)
#
# img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(),face.bottom()),color=(255,255,255),thickness=1)
# img = cv2.resize(img,dsize=(48,48)).astype(np.float64)
#
# pred = model.predict(img.reshape(1,48,48,1))
# print(pred)
#
# plt.subplot(1,1,1)
# plt.imshow(img)
# plt.show()


# cv2.imshow('test',img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

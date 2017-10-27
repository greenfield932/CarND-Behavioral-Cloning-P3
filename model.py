import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Activation
from keras import backend as K
import matplotlib.pyplot as plt

lines = []
fd = open('../data/record/driving_log.csv')
reader = csv.reader(fd)

cnt = 0
for line in reader:
    if cnt > 0:
        lines.append(line)
    cnt+=1

center_img_idx = 0
left_img_idx = 1
right_img_idx = 2
steering_idx = 3
throttle_idx = 4
break_idx = 5
speed_idx = 6

steering_data = []
image_data = []

correction = 0.1 # this is a parameter to tune

for data in lines:
    image = cv2.imread(data[center_img_idx])
    image_data.append(image)
    steering_center = float(data[steering_idx])
    steering_data.append(steering_center)
    if len(steering_data) > 5:
        break
    #image_left = cv2.imread(data[left_img_idx])
    #steering_left = steering_center + correction
    #image_data.append(image_left)
    #steering_data.append(steering_left)
    
    #image_right = cv2.imread(data[right_img_idx])
    #steering_right = steering_center - correction
    #image_data.append(image_right)
    #steering_data.append(steering_right)

X_train = np.array(image_data)
y_train = np.array(steering_data)

model = Sequential()
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

cropping_output = K.function([model.layers[0].input], [model.layers[0].output])
image = image_data[0]
cropped_image = cropping_output([image[None,...]])[0]

f, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(cropped_image)
plt.show()
#cropped_image = cropping_output([image[None,...]])[0]
#model.add(Flatten(input_shape = (90, 320, 3)))
#model.add(Convolution2D(24, 5, 5, border_mode='same', output_shape=(24,45,160)))
#model.add(Activation('relu'))
#model.add(Convolution2D(36, 5, 5, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(img_rows/2, img_cols/2)))
#model.add(Flatten(input_shape = (90, 320, 3)))
#model.add(Dense(1))

#model.compile(loss = 'mse', optimizer = 'adam')
#model.summary()
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 15)

#model.save('model.h5')



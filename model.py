import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Activation
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import sklearn

def readInputData(driving_log_filename):
    lines = []
    fd = open(driving_log_filename)
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

    samples = []

    correction = 0.35 # this is a parameter to tune
    
    for data in lines:
        #image = cv2.imread(data[center_img_idx])
        #image = Image.open(data[center_img_idx])
        #image = np.asarray(image)

        steering = float(data[steering_idx])
        
        samples.append([data[center_img_idx], steering])

        samples.append([data[left_img_idx], steering + correction])

        samples.append([data[right_img_idx], steering - correction])

    return samples


def drawImage(image):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

def drawImages(images, rows, cols, labels = []):

    fig = plt.figure(figsize = (cols*4,rows*2))
#    plt.subplots_adjust(wspace = 0.3, hspace = 1)
    
    for i in range(0, len(images)):
        image = images[i]
        subplot = fig.add_subplot(rows, cols, i+1)
        subplot.imshow(image)
        if len(labels)>0:
            subplot.set_title(labels[i])
        subplot.axis('off')
    plt.show()
    
    #image_left = cv2.imread(data[left_img_idx])
    #steering_left = steering_center + correction
    #image_data.append(image_left)
    #steering_data.append(steering_left)
    
    #image_right = cv2.imread(data[right_img_idx])
    #steering_right = steering_center - correction
    #image_data.append(image_right)
    #steering_data.append(steering_right)

def evalLayer(model, image, idx):
    cropping_output = K.function([model.layers[idx].input], [model.layers[idx].output])
    return cropping_output([image[None,...]])[0]

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

def augment(image, steering, augtype):
    if augtype == 'flip':        
        image_flipped = np.fliplr(image)
        steering_flipped = -steering
        return image_flipped, steering_flipped
    elif augtype == 'shift':
        return trans_image(image, steering, 30)
    
    return image, steering
        
def generator(samples, batch_size=32):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = Image.open(batch_sample[0])
                image = np.asarray(image)
                steering = batch_sample[1]
                
                images.append(image)
                angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


samples = readInputData('../data/record/driving_log.csv')
samples = np.array(samples)

train, val = train_test_split(samples, test_size=0.2)

train_generator = generator(train, batch_size=32)
val_generator = generator(val, batch_size=32)

n_train = len(train)
n_val = len(val)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
#model.add(MaxPooling2D(pool_size=(img_rows/2, img_cols/2)))
#cropping_output = K.function([model.layers[0].input], [model.layers[0].output])
#image = image_data[0]
#cropped_image = cropping_output([image[None,...]])[0]
#cropped_image = np.uint8(cropped_image[0,...])
#drawImage(cropped_image)

model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(96, 3, 3, subsample=(2, 2), border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

#f, ax = plt.subplots(1, 1, figsize=(5, 5))
#ax.imshow(cropped_image)
#plt.show()
#cropped_image = cropping_output([image[None,...]])[0]
#model.add(Flatten(input_shape = (90, 320, 3)))

#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',  input_shape=(3,160,320)))
#model.add(Activation('relu'))

#model.add(Convolution2D(36, 5, 5, border_mode='same'))
#model.add(MaxPooling2D(pool_size=(img_rows/2, img_cols/2)))
#model.add(Flatten())
#model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

#model.summary()
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 8)

model.fit_generator(train_generator, samples_per_epoch=n_train,
                    validation_data=val_generator, nb_val_samples=n_val,
                    nb_epoch=3)

model.save('model.h5')

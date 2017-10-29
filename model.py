import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Activation
from keras import backend as K #for crop layer visualisation
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import sklearn
from keras.utils.visualize_util import plot #for model graph visualization purposes

#read driving log: images filenames and steering values
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
    
    cnt = 0
    for data in lines:
        steering = float(data[steering_idx])
        samples.append([data[center_img_idx], steering])
        samples.append([data[left_img_idx], steering + correction])
        samples.append([data[right_img_idx], steering - correction])
        if cnt > 22000:
            break
        cnt+=1
    return samples

# read image data from file
def readImage(filename):
    image = Image.open(filename)
    return np.asarray(image)

# draw image via matplot
def drawImage(image):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

# draw multiple images via matplot
def drawImages(images, rows, cols, labels = []):

    fig = plt.figure(figsize = (cols*4,rows*2))   
    for i in range(0, len(images)):
        image = images[i]
        subplot = fig.add_subplot(rows, cols, i+1)
        subplot.imshow(image)
        if len(labels)>0:
            subplot.set_title(labels[i])
        subplot.axis('off')
    plt.show()

# augment data by flipping and brighthness
def augment(image, steering, augtype):
    if augtype == 'flip':        
        image_flipped = np.fliplr(image)               
        steering_flipped = -steering
        return image_flipped, steering_flipped
    elif augtype == 'bright':
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype = np.float32)
        random_bright = .5+np.random.uniform()
        image[:,:,2] = image[:,:,2]*random_bright
        image[:,:,2][image[:,:,2]>255]  = 255
        image = np.array(image, dtype = np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
        return image, steering    
    return image, steering

# augmentation multiplier - adjust samples size depending on augmentation techniques amount
augment_multiplier = 2

# data generator
def generator(samples, batch_size=32*augment_multiplier):

    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/augment_multiplier)):
            batch_samples = samples[offset:offset+int(batch_size/augment_multiplier)]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                image0 = readImage(batch_sample[0])
                steering = float(batch_sample[1])
                
                image = image0
                
                images.append(image)
                angles.append(steering)
                
                image, steering = augment(image0, steering, 'flip')
                images.append(image)
                angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# for report purposes - draw original and flipped image
def showFlip(samples):
    idx = 99
    img0 = readImage(samples[idx][0])
    steer = float(samples[idx][1])
    img1, steer_flip = augment(img0, steer, 'flip')
    drawImages([img0, img1], 1, 2, ["original: {0:.2f}".format(steer) , "flipped: : {0:.2f}".format(steer_flip)] )

# for report purposes - draw left/center/right images and steering angles
def showCorrection(samples):
    idx = 0
    img_center = readImage(samples[idx][0])
    img_left = readImage(samples[idx+1][0])
    img_right = readImage(samples[idx+2][0])

    steer_center = "{0:.2f}".format(float(samples[idx][1]))
    steer_left ="{0:.2f}".format(float(samples[idx+1][1]))
    steer_right ="{0:.2f}".format(float(samples[idx+2][1]))

    drawImages([img_left, img_center, img_right], 1, 3, ['left_image:'+steer_left, 'center_image:'+steer_center, 'right_image:'+steer_right])

# for report purposes - draw cropped image using keras lambda layer
def showCrop(model, samples):
    image = readImage(samples[0][0])
    cropping_output = K.function([model.layers[0].input], [model.layers[0].output])
    cropped_image = cropping_output([image[None,...]])[0]
    cropped_image = np.uint8(cropped_image[0,...])
    drawImages([image, cropped_image],1,2,['original','cropped'])

# for report purposes - draw model graph  
def drawModel(model):
    plot(model, to_file='model.png',show_shapes=True, show_layer_names=True)  

# draw train/val loss plot
def drawLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def LeNet():
    
    model = Sequential()

    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3))) #crop image to remove non relevant data (sky and car head)

    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # normalize data (mean = 0)

    
    model.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode='valid')) #first convolutional layer 6 filters, ReLU activation, max pooling
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode='valid')) #second convolutional layer 16 filters, ReLU activation, max pooling
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Flatten()) # Flatten
    model.add(Dense(400)) #Fully connected 1
    model.add(Dense(120)) #Fully connected 2
    model.add(Dense(84)) #Fully connected 3 
    model.add(Dense(1)) #Fully connected 4 
    
    return model

def DAVE():
    model = Sequential()

    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3))) #crop image to remove non relevant data (sky and car head)

    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # normalize data (mean = 0)
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid')) 
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid')) 
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid')) 
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid')) 
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))


    model.add(Flatten()) # Flatten
    model.add(Dense(100)) #Fully connected 1
    model.add(Dense(50)) #Fully connected 2
    model.add(Dense(10)) #Fully connected 3 
    model.add(Dense(1)) #Fully connected 4 
    
    return model
    
    
def MyNet():
    model = Sequential()

    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3))) #crop image to remove non relevant data (sky and car head)

    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # normalize data (mean = 0)
    
        
    model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='valid')) #first convolutional layer 24 filters, pooling, dropout, ReLU activation
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))  #second convolutional layer 48 filters, pooling, dropout, ReLU activation
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3, subsample=(2, 2), border_mode='valid'))  #third convolutional layer 96 filters, pooling, dropout, ReLU activation
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Flatten()) # Flatten
    model.add(Dense(128)) #Fully connected 1
    model.add(Dense(64)) #Fully connected 2
    model.add(Dense(1)) #Fully connected 3
    
    return model    

samples = readInputData('../data/record/driving_log.csv')
samples = np.array(samples)
train, val = train_test_split(samples, test_size=0.2)
#train, val = train_test_split(samples, test_size=0.25)
#train, val = train_test_split(samples, test_size=0.3)

train_generator = generator(train, batch_size=32*augment_multiplier)
val_generator = generator(val, batch_size=32*augment_multiplier)
n_train = len(train)*augment_multiplier
n_val = len(val)*augment_multiplier

#CNN model definition
model = MyNet()

model.compile(loss = 'mse', optimizer = 'adam')

#drawModel(model)
model.summary()
history = model.fit_generator(train_generator, samples_per_epoch=n_train,
                    validation_data=val_generator, nb_val_samples=n_val,
                    nb_epoch=10)
drawLoss(history)
model.save('model.h5')
#showCorrection(samples)
#showFlip(samples)
#showCrop(model, samples)
#drawImages([readImage(samples[0][0])],1,1,['center image'])


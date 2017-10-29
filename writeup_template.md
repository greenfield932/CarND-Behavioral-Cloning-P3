# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image3]: ./examples/flip.png "Flip Image"
[image4]: ./examples/correction.png "Correction Image"
[image5]: ./examples/crop.png "Crop Image"
[image6]: ./examples/loss.png "Loss"
[image7]: ./examples/driving_example.png ""



## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 96 (model.py lines 221-244) 

The model includes RELU layers to introduce nonlinearity (code lines 229, 234, 239), and the data is normalized in the model using a Keras lambda layer (code line 223). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 228, 233, 238). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 250-252). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 262).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving with left/right/center camera image data. Left and right images were used to 
train model for recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to move from simple network architecture to more complex network.

My initial solution was a LeNet network with 2 convolutional layers, ReLU activation, max pooling and 4 fully connected layers with total of ~9.5 millions of parameters on 320x90 input image. The network trained well except first epochs (train loss 1.35) with training loss around 0.0037 and validation loss about 0.0069 on the 8-th epoch. Loss decreased smoothly the whole training time after first epoch. The trained network was able to drive the whole track correctly, but the car went too close to the right side of the bridge.

After that I tried NVIDIA DAVE network. It was also able to drive the whole track correctly after 8 epochs except shifting to the left side of the road near the lake (road turning right), but after the turn it was successfully recovered to the center of the road. Training results were about 0.01 on training set and 0.026 on testing at the 8-th epochs, testing loss increased constantly on all epochs while training loss decreasing. It looks like overfitting so I decided to modify the network and add dropout layers.

I also experemented with amount of layers. I removed 2 of 3 convolutional layers and got approximately the same results while model consists of 300,000 parameters instead of ~1 million I got with DAVE network and 320x90 input image. Loss remain approximately similar, but steering angles became much more smooth.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 219-246) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

My training data consists of 2 laps of normal driving in the center of the road. Example of the image from central camera:

![alt text][image7]

To train model of how to recover from from the left and right sides of the road I used images from left and right cameras extended with angle correction.
Thus, angle was corrected +0.35 for left camera image and -0.35 for right camera image. The correction parameter was found experimentally. Example of images and angles correction:

![alt text][image4]

To augment the dataset, I also flipped images and angles so the data of how to drive right and left will be better balanced. So the model will better generalize of how to drive in both directions, not only one, while track data was recorded for one direction only. For example, here is an image that has then been flipped:

![alt text][image3]

To remove non-relevant data for driving from images (sky and car head) I used keras lambda layer to crop input images. Example of original and cropped images:

![alt text][image5]

After the collection process, I had 9126 number of data points. I then preprocessed this data by flipping and got 18252 samples.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the training/validation plot

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

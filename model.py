###################################
# SCRIPT - LIBRARY IMPORTS
###################################

# Script management
import os
import sys
import time

# Data processing and visualization
import random
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

###################################
# VARIABLES
###################################

# Paths
img_csvPath = 'data/driving_log.csv'
img_jpgPath = 'data/IMG/'

# Model parameters
n_epochs = 5
n_batch = 256

# Script Performance
time_total = time.time()

###################################
# FUNCTIONS
###################################

# Vectorize dataset
def vectorizeData(dat_csv, correction=0.2, null_thres=0.001):
	'''
	takes input image data and splits it into two lists: image paths, angles
	'''
	dat_filepath = []
	dat_angle = []
	for data in dat_csv:
		# check that angle above threshold
		if abs(float(data[3])) > null_thres:
			for i in range(3):
				if i == 0:
					correction_angle = 0
				else:
					correction_angle = -1 ** (i-1) * correction
				temp_path = data[i]
				# append data images and angles to lists
				dat_filepath.append(img_jpgPath + temp_path.split('/')[-1])
				dat_angle.append(float(data[3]) + correction_angle)
	# convert lists into numpy arrays
	dat_filepath = np.array(dat_filepath)
	dat_angle = np.array(dat_angle)
	return (dat_filepath, dat_angle)

# Visualize dataset
def visualizeData(dat_angles):
	'''
	takes input array of image angles and outputs the distribution in a histogram
	'''
	n_bins = 21
	angles_avg = len(dat_angles) / n_bins
	hist, bin_edges = np.histogram(dat_angles, bins=n_bins)
	width = 0.5 * (bin_edges[1] - bin_edges[0])
	center = (bin_edges[:-1] + bin_edges[1:]) / 2
	plt.bar(center, hist, align = 'center', width = width)
	#show average line
	#plt.plot((np.min(dat_angles), np.max(dat_angles)), (angles_avg, angles_avg), 'k-')
	#plt.show()
	plt.savefig('images/anglesHist.png')
	return (hist, bin_edges)

# Equalize dataset
def equalizeData(dat_paths, dat_angles, hist, bin_edges, keep_thres):
	'''
	remove samples from a bin down to the keep_thres value
	'''
	dat_paths, dat_angles = shuffle(dat_paths, dat_angles, random_state=0)
	dat_paths_out = []
	dat_angles_out = []
	hist_count = np.zeros(len(hist))
	for i in range(len(dat_angles)):
		# check which bin the angle falls under
		for j in range(len(hist)):
			if (dat_angles[i] > bin_edges[j]) and (dat_angles <= bin_edges[j+1]):
				# append filepath and angle if the hist count is within threshold
				if hist[j] <= keep_thres:
					hist_count += 1
					dat_paths_out.append(dat_paths[i])
					dat_angles_out.append(dat_angles[i])
				# if hist count is above threshold, omit sample
				continue
	dat_paths_out = np.array(dat_paths_out)
	dat_angles_out = np.array(dat_angles_out)
	return (dat_paths_out, dat_angles_out)


# Generator
def generator(dataset, batch_size = 32, threshold = 0.3):
    #print(dataSet)
    filenames, angles = dataset
    n_samples = len(angles)


    ##########################
    #initialize temp arrays
    image_out = []
    angle_out = []
    while True:
        #generator loop forever
        filenames, angles = shuffle(filenames, angles)
        for i in range(n_samples):
            #read image and convert to RGB so it works on video.py
            img = cv2.imread(filenames[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #opportunity for more preprocessing here
            image_out.append(img)
            angle_out.append(angles[i])

            if len(image_out) == batch_size:
                #When we fill the batch, sent it out
                X_train = np.array(image_out)
                y_train = np.array(image_out)
                #empty temp arrays
                image_out = []
                angle_out = []
                yield shuffle(X_train, y_train)

            if abs(angles[i]) > threshold:
                #create a flipped version if the angle is above a certain threshold
                image_out.append(cv2.flip(img, 1))
                angle_out.append(-1.0 * angles[i])

                if len(image_out) == batch_size:
                    #When we fill the batch, sent it out
                    X_train = np.array(image_out)
                    y_train = np.array(angle_out)
                    #empty temp arrays
                    image_out = []
                    angle_out = []
                    yield shuffle(X_train, y_train)

###################################
# DATA - IMPORTING
###################################

# initialize list to store data
img_paths = []
# open csv file and append data to list
with open(img_csvPath) as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		img_paths.append(line)

###################################
# DATA - PROCESSING
###################################

# vectorize data
img_dataP = vectorizeData(img_paths)
# visualize data distribution
img_data_visual = visualizeData(img_dataP[1])
'''
for i in range(len(img_dataP_visual[0])):
	print("range [{}, {}]: {}".format(img_dataP_visual[1][i],img_dataP_visual[1][i+1],img_dataP_visual[0][i]))
'''
# equalize data using preprocessed data and its visualization
img_dataPEq = equalizeData(img_dataP, img_data_visual)
# visualize data distribution of equalized data
img_dataPEq_visual = visualizeData(img_dataPEq[1])

X_train, X_valid, y_train, y_valid = train_test_split(img_dataP[0], img_dataP[1], test_size=0.2)

train_generator = generator((X_train, y_train), batch_size = n_batch, threshold = 0)
validation_generator = generator((X_valid, y_valid), batch_size = n_batch, threshold = 0)


###################################
# MODEL - KERAS
###################################

# Keras Modeling
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(64, (3,3), activation="elu"))
model.add(Conv2D(64, (3,3), activation="elu"))
'''
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
'''
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.save('model.h5')
print('Model Saved')

history_object = model.fit_generator(train_generator, samples_per_epoch = len(X_train), validation_data = validation_generator, nb_val_samples = len(X_valid), nb_epoch=n_epochs, verbose = 1)

###################################
# MODEL - PERFORMANCE
###################################

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

###################################
# SCRIPT - CLOSING
###################################
exit()

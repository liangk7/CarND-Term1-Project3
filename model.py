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
		# check that angle is above threshold
		if abs(float(data[3])) > null_thres:
			for i in range(3):
				if i == 0:
					correction_angle = 0
				else:
					correction_angle = (-1) ** (i-1) * correction
				temp_path = data[i]
				# append data images and angles to lists
				dat_filepath.append(img_jpgPath + temp_path.split('/')[-1])
				dat_angle.append(float(data[3]) + correction_angle)
	# convert lists into numpy arrays
	dat_filepath = np.array(dat_filepath)
	dat_angle = np.array(dat_angle)
	return (dat_filepath, dat_angle)

# Equalize dataset
def equalizeData(dataset):
	dat_paths, dat_angles = shuffle(dataset[0], dataset[1], random_state=0)
	n_bins = 25
	angles_avg = len(dat_angles) / n_bins
	# visualize histogram of data
	hist, bin_edges = np.histogram(dat_angles, bins=n_bins)
	width = 0.8 * (bin_edges[1] - bin_edges[0])
	center = (bin_edges[:-1] + bin_edges[1:]) / 2
	#plt.bar(center, hist, align='center', width=width, color='b', label='raw data')
	# show average line
	#plt.plot((np.min(dat_angles), np.max(dat_angles)), (angles_avg, angles_avg), 'k-')
	# create output lists of filepaths and angles
	dat_paths_out = []
	dat_angles_out = []
	hist_count = np.zeros(len(hist))
	for i in range(len(dat_angles)): # check which bin the angle falls under
		for j in range(len(hist)):
			if (dat_angles[i] > bin_edges[j]) and (dat_angles[i] <= bin_edges[j+1]):
			# append filepath and angle if the hist count is within threshold
				if hist_count[j] <= angles_avg:
					hist_count[j] += 1
					dat_paths_out.append(dat_paths[i])
					dat_angles_out.append(dat_angles[i])
	dat_paths_out = np.array(dat_paths_out)
	dat_angles_out = np.array(dat_angles_out)
	# visualize histogram of adjusted data
	#plt.bar(center, hist_count, align='center', width=0.5*width, color='r', alpha=0.5, label='adjusted data')
	#plt.savefig('images/plot_anglesHist.png')
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
		filenames, angles = shuffle(filenames, angles)
		for i in range(n_samples):
			#read image and convert to RGB
			img = cv2.imread(filenames[i])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			image_out.append(img)
			angle_out.append(angles[i])
			# add a mirrored data sample if the angle is above a certain threshold
			if len(image_out) < batch_size and abs(angles[i]) > threshold:
				image_out.append(cv2.flip(img, 1))
				angle_out.append(-1.0 * angles[i])
			# when batch size is achieved, push the dataset
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
# equalize data using preprocessed data and its visualization
img_dataPEq = equalizeData(img_dataP)

X_train, X_valid, y_train, y_valid = train_test_split(img_dataPEq[0], img_dataPEq[1], test_size=0.2)

train_generator = generator((X_train, y_train), batch_size = n_batch, threshold = 0)
validation_generator = generator((X_valid, y_valid), batch_size = n_batch, threshold = 0)


###################################
# MODEL - KERAS
###################################

# Keras Modeling
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch = len(X_train), validation_data = validation_generator, nb_val_samples = len(X_valid), nb_epoch=n_epochs, verbose = 1)

model.save('model.h5')
print('Model Saved')

###################################
# MODEL - PERFORMANCE
###################################
'''
plt.gcf().clear()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Squared Error vs Epochs')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('images/plot_errorLoss.png')
plt.ion()
plt.show()
'''
###################################
# SCRIPT - CLOSING
###################################
exit()

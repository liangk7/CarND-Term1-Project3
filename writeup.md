# **Behavioral Cloning**

---

**Developing a Autonomous Driving Model using Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./images/figure_nvidiaCNNarchitecture.png "NVIDIA architecture"
[image2]: ./images/plot_anglesHist.png "Angle Equalization"
[image3]: ./images/plot_errorLoss.png "MSE vs Epochs"

#### Sources
Udacity SDC Nanodegree: [CarND-Behavioral-Cloning-P3](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
[Simulator](https://github.com/udacity/self-driving-car-sim)

---

## Writeup / README

This is the writeup for my Udacity Self-Driving Car Nanodegree Term 1 [Project 3 submission](https://github.com/liangk7/CarND-Term1-Project3) in accordance with the [rubric guidelines](https://review.udacity.com/#!/rubrics/432/view)

---

### Required Files

#### Files submitted
- `model.py`:	utilizes a dataset of images and angles to create `model.h5`
- `drive.py`:	demonstrates performance of `model.h5` in the **simulator**
- `model.h5`:	file that contains parameters of the autonomous driving model
- `writeup.md`:	contains the information of project design and usage
- `video.mp4`:	animation that demonstrates the `model.h5` performance

---

### Code Structure

#### Code Function
As outlined in the project skeleton, `model.py` uses data from the `driving_log.csv` file and `IMG/` folder to create a model `model.h5`. This model is then demoed in the **simulator's** *automatic* setting using `drive.py`.

#### Code Usage
In order to complete the project, one must follow these steps:
1) Access the **simulator** (listed as a source)
- choose a track and select *Training Mode*
- record training data to a known folder directory
2) In order to develop a `model.h5` file, one must update and run the `model.py` file
- install necessary libraries for python
- change *Paths* variables to directories of recorded data (as needed)
3) To evaluate model performance, run `drive.py`
- be sure to pass `model.py` as the model `python drive.py model.h5`
- open the **simulator** and select *Autonomous Mode*

---

### Model Architecture and Training Strategy

#### Model architecture
The general structure of `model.py` incorporates data normalization and sample equalization. When getting to the convolutional neural network, the model (using  keras layers) employed is the Nvidia CNN architecture. 

![alt text][image1]


#### Reducing overfitting
By observing the data, one may notice that there is a large bias towards the angle measurement of `0`. Although `0` is a common, and often desirable, angle value (especially on straightaways), the likelihood of its substantial repetition may also stem from error in the simulation dataset. For example, during the development of data, there may be instances (on a curve) that the user turns too sharply and must compensate by counter-steering. Such action could result in a `0` angle measurement (or even a measurement of opposite polarity). Thus, with the use of linear regression, it may ultimately be more beneficial to limit these types of errors by removing the `0` angle measurements altogether.
Relative to this occurrence, it is obvious that most angle measurements tend towards smaller magnitude values (closer to 0). Thus, to prevent a large bias in the linear regression function, equalizing the dataset across histogram bins proves to be a valuable technique in developing an accurate model.

#### Parameter tuning
The tunable parameters outside of the convultional neural network are:
- epochs: increase of this parameter will result in a convergence of error loss between training and validation datasets. In the case of the Nvidia CNN model, the choice of 5 epochs yielded a sufficient result for simulation performance
- batch size: increase of this parameter will result in an inability to generalize which produces a lower quality model.

#### Training data
Since the training data is a compilation of image trios gathered 


---

### Architecture and Training Documentation

#### Solution performance


#### Model architecture

|  Layer			|  Description					|
|:------------------|------------------------------:|
|  Input			|  160x320x3 RGB image			|
|  Normalization 	|  [0,255] to [0,1]				|
|  Convolution 5x5	|  2x2 stride, VALID padding,	|
|					|	outputs 62x196x24			|
|  ELU 				|								|
|  Max pooling		|  outputs 24@31x98				|
|  Convolution 5x5	|  2x2 stride, VALID padding,	|
|					|	outputs 36x28x94			|
|  ELU 				|								|
|  Max pooling		|  outputs 36@14x47				|
|  Convolution 5x5	|  2x2 stride, VALID padding,	|
|					|	outputs 48@10x44			|
|  ELU 	 			|								|
|  Max pooling		|  outputs 48@5x22				|
|  Convolution 3x3	|  1x1 stride, VALID padding,	|
|					|	outputs 64@3x20				|
|  ELU 	 			|								|
|  Convolution 3x3	|  1x1 stride, VALID padding,	|
|					|	outputs 64@1x18				|
|  ELU 	 			|								|
|  Flatten			|  64@1x18 -> 1152				|
|  Fully Connected	|  outputs 100					|
|  Dropout			|  keep_probability: 0.5 		|
|  Fully Connected	|  outputs 50					|
|  Fully Connected	|  outputs 10					|
|  Fully Connected	|  outputs 1					|


#### Dataset creation
The creation of a dataset is performed using the **simulator** (referenced in sources). To acquire a set of images that encompasses a wide range of sample data, I drove the entire track, in both directions, bearing left and right towards the borders.

![alt text][image2]

---

### Simulation

#### Is the car able to navigate correctly on test data?



---

### Future Implementation


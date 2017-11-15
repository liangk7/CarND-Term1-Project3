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
[image3]: ./images/ss_modelLoss.png "Model Loss"
[image4]: ./images/plot_errorLoss.png "MSE vs Epochs"

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

By visualizing the dataset, we can determine what measures to take to prevent overfitting.

![alt text][image2]

Paying attention to the raw dataset (shown as blue bars), we can note a bias around the `0` angle as well as small angles (near magnitude `0.2`). To prevent any bearing towards these specific angles, an equalization measure is taken based on the average number of data points per histogran bin (shown as red bars). 

#### Parameter tuning
The tunable parameters outside of the convultional neural network are:
- epochs: increase of this parameter will result in a convergence of error loss between training and validation datasets. In the case of the Nvidia CNN model, the choice of 5 epochs yielded a sufficient result for simulation performance
- batch size: increase of this parameter will result in an inability to generalize which produces a lower quality model. The default option of `256` was selected and proved sufficient for simulation performance

#### Training data
Since the training data is a compilation of image trios, the amount of data captured in each lap provides a diverse image pool in which the model can be trained. However, attempts at building a model only with centered lap data proved unsuccessful, with many off-road (water and dirt) gradients being misclassified as drivable routes. To enhance the proper classification of these scenarios, additional data was taken at these segments of the track.


---

### Architecture and Training Documentation

#### Solution performance


![alt text][image3]

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


![alt text][image4]

#### Dataset creation
The creation of a dataset is performed using the **simulator** (referenced in sources). To acquire a set of images that encompasses a wide range of sample data, I drove the entire track, in both directions, bearing left and right along the borders near large gradients (water and dirt paths).


---

### Simulation

#### Is the car able to navigate correctly on test data?
As demonstrated in the `video.mp4` file, `model.h5` was able to maneuver the car around the track for a full lap without crossing the lane lines. But while the car does remain in the designated path, it tends to utilize an approach that is more corrective than predictive. And with the addition of data that adheres to boundaries and makes hard corrections away from them, the simulated car has a wavering path rather than a smooth curved path.


---

### Closing Thoughts

#### Conclusion
While many of the techniques utilized in this project were insightful, the simulation clearly demonstrated that a generalized approach towards computer vision using convolutional neural networks on the full environment (with just optical cameras) would not be sufficient in detecting road hazards and compensating for various vehicle orientations.

#### Future Implementation
After coming to this conclusion, it seems rather unlikely that behavioral cloning would be a cornerstone in the practical deployment of autonomous vehicles. However, in terms of developing a good baseline model of a controlled environment, it may actually hold some value. And if we were to use behavioral cloning in a real-time scenario involving a mixed autonomous/human operator environment, it would be possible for an autonomous vehicle to then predict how a human driver in front of it will maneuver and respond to potential hazards.

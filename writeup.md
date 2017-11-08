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

[image1]: ./example_path "example_text"
[image2]: ./example_path "example_text"

#### Sources
Udacity SDC Nanodegree: [CarND-Behavioral-Cloning-P3](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
[Simulator](https://github.com/udacity/self-driving-car-sim)

---

## Writeup / README

This is the writeup for my Udacity Self-Driving Car Nanodegree Term 1 [Project 3 submission](https://github.com/liangk7/CarND-Term1-Project3) in accordance with the [rubric guidelines](https://review.udacity.com/#!/rubrics/432/view)

---

### Required Files

#### Files submitted
- `model.py`
- `drive.py`
- `model.h5`
- `writeup.md`
- `video.mp4`
![alt text][image1]

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


#### Reducing overfitting


#### Parameter tuning


#### Training data



---

### Architecture and Training Documentation

#### Solution performance


#### Model architecture


#### Dataset creation



---

### Simulation

#### Is the car able to navigate correctly on test data?



---

### Future Implementation


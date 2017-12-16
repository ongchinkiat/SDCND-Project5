## Self-Driving Car Engineer Nanodegree

## Project 5: Vehicle Detection and Tracking

## Introduction

The goals / steps of this project are the following:


* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Project Files:

* Notebook File: vehicle.ipynb
* Notebook HTML: vehicle.html
* Project Video: project_video_out.mp4

## Histogram of Oriented Gradients (HOG)

### Data Set

The vehicle and non-vehicle data set provided by Udacity is used to train the classifier. The data set images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.

There are a total of 17760 images:

* Cars Images: 8792
* Non-Cars Images: 8968

The images are all in size 64x64.

This is an example of Car and Non-Car images:

![Car Non-car](https://github.com/ongchinkiat/SDCND-Project5/raw/master/car-notcar.png "Car Non-car")

### Features Extraction

For each image, features are extracted for the training of the classifier.

After some experimentation, these parameters gives us the best accuracy result for the classifier:

* Spatial features
  * Spatial Size = (32, 32)
* Color Histogram
  * RGB Color Space
  * 32 bins for each color channel
* Histogram of Oriented Gradients (HOG)
  * RGB Color Space
  * Orientation: 9
  * Pixels Per Cell: 8
  * Cells Per Block: 2


### Training Of classifier

The feature sets are randomised and then split into Training and Testing set, at a ratio of 80% Training set, 20% Testing set.

A C-Support Vector Classifier with RBF kernel is used to train the classifier.

The resulting SVC gives an accuracy score of 99.5%.

## Video Implementation

To speed up the process of search for cars in the video images, we limit the search to the region of y = 400 to y = 656, ignoring the skyline region and the area too near to our car.

Sliding Window search was also used to speed up processing. Since the calculation of HOG features is slow, HOG features for the entire image is calculated only once, and stored as an array.

```
hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```

For each prediction block, the relevant HOG cells were recalled from the stored array, and combined with the spatial and color histogram features. This helps to speed up processing significantly.

To reduce false positives, heat maps are used to consolidate the prediction results. For each positive prediction, the region of the predicted square is increased by 1. Since we are using a sliding window search, areas where a vehicle is tend to generate positive results multiple times. By applying a detection threshold, we can filter out areas which only trigger false positives a couple of times.

This is a visualization of the heat map for a single image:

![Heat Map](https://github.com/ongchinkiat/SDCND-Project5/raw/master/heatmap.png "Heat Map")

To further reduce false positives, the heat maps of previous images were retained and combined with the current image heat map. A higher detection threshold was also used on the combined heat map.


```
    detect_threshold = 15

    hmaps[framenum % 10] = thisheatmap
    framenum += 1

    addheatmap = np.zeros_like(img[:,:,0])
    for h in hmaps:
        addheatmap = cv2.add(addheatmap, h)

    # Zero out pixels below the threshold
    addheatmap[addheatmap <= detect_threshold] = 0
```

After some experimentation, I used these parameters:
* Number of history frames = 10
* Detection Threshold = 15


The resulting processed video is in the file: project_video_out.mp4

## Discussion

In this project, we have successfully implemented a Vehicle Detection and Tracking algorithm to detect cars in a video stream.

Support Vector Machine algorithm is used for the car / not car classification of the video images.

Better performance may be achieved if we use Convolutional Neural Network as classifier, but it need much more computing power.

For example, I could have use a fully convolutional network (FCN) which I had implemented in the Udacity Robotics Project 4 - Follow Me:

https://github.com/ongchinkiat/robond-follow-me

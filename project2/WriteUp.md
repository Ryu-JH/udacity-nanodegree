## Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./output_images/ex_dist.png "Distorted"
[image2]: ./output_images/ex_disted.png "Undistorted"
[image3]: ./output_images/dist.png "Distorted"
[image4]: ./output_images/disted.png "Undistorted"
[image5]: ./output_images/raw.png "Fit Visual"
[image6]: ./output_images/bird.png "Output"
[video1]: ./project_video_output.mp4 "Video"


### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced_Lane_Finding.ipynb".  

I start by preparing `objpoints`, which will be the (x, y, z) coordinates of the chessboard corners in the world.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. 'objp' created by np.mgrid function is same in every image.`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. To find `imagepoints` ,the image was converted into grayscale and it was fed to `cv2.findChessboardCorners` functions.   Then, I used the 'objpoints' and 'imgpoints' into 'cv2.calibrateCamera' function to compute camera matrix and distortion coefficients. And by using the 'cv2.undistort' function, I applied this distortion correction to the test image. The result is 
![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

The following image is in the ipynb file

#### 2. Color transforms, gradients or other methods to create a thresholded binary image. 
The codes explained below are about 'helper.py'.
'helper.Binary' function creates a binary image to find lanes easily.  
This function use both HLS and Gray scale transformation that was used in project 1.
Now 'cv2.getPerspectiveTransform' and 'cv2.warpPerspective' function used to make bird_eye view.
The result is 
![alt text][image3]
![alt text][image4]

#### 4. Identifing lane-line pixels and fitting their positions with a polynomial.
I imported 'from line import Line' to find left and right lane points.
The 'line' object is called with an image input. It help me to get corrects for distortion, warps in to bird eye view and find binary pixel of interests. The result is
![alt text][image5]
![alt text][image6]

---

### Pipeline (video)

#### 1. link to the final video output. 

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

Well finding lanes and curvature with current methd fails for the challenge videos. [in this video](challenge_binary.mp4) shows that this methods can't classfy the target lane and the other lane. It should be applied more advanced techniques like deep learing.
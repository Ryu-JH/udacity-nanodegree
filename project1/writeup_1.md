# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps.
1. gray_scale
Change the image to grayscale with `grayscale` function. It makes me much easier to find edges in grayscale.
2. gaussian blur
Use `gaussian_blur` function with `kernel_size=5` to blur out noise. 
3. canny edge detection
Apply `canny` with `low_threshold=50` and `high_threshold=150`. With this parameters canny edge detection found clean edges.
4. Region of interest
ROI function grap the small region to find lane easily. By applying ROI, I can apply other functions to what i want. Because the lane lines are almost always found in small region of the image. `region_of_interest` function cuts out everywhere else except for this small region so that I can find only our lane. My region of interest is defined as a trapezoid. It's height is only 2/5 of the image height. The bottom length and top length of the trapezoid is set to image width and 40 pixels respectively.
5. finding HoughLines
Using cv2.HoughLinesP with `rho = 2`, `theta = np.pi/180`, `threshold=30`, `min_line_len=100`, `max_line_gap=200` to overlay detected lanes on top of the original image. I've found that calling HoughLinesP with these parameter give me satisfying result.
6. draw_lines
Draw the detected lines finally. I tried to denote my right and left lanes using different colors thus it make easier to understand by identifying both the lanes individually. This works well when drawing lanes over videos.


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming is that it doesn't generalize well about curved lanes.
And it could detect when a given line segment has both of left and right.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be using deep learning techniques if there are enough datas to detect lanes. 
Another potential improvement could be to use multi lane detection argorithmn. By comparing their result, we can select the results.

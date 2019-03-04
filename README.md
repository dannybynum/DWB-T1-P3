# Danny's Udacity Term1 Project:  Build a Traffic Sign Classifier (using a CNN)

## Background / Scope of project
This project was in the form of a Jupyter notebook (similar format to project 1).  A really good starting place was provided from the LeNet lab notebook.  This notebook has some really good starting code in it an the course also provides videos to walk through the code line-by-line as well as including a video that describes what may have to change to adapt the lab notebook code for the project.

However, it was also subtly difficult because in order to achieve all of the "rubric points" you really did have to understand and apply many of the concepts from all of the lessons leading up to the project.  One good example of this is with having to actually show the model prediction for new images from the web.  The code given provides a good pipeline for training the images but you have to dig into how TensorFlow works in order to extract the prediction by itself outside of the training model.

---

**The fundamental steps required were as follows:**
* Load the data set (after downloading from given site)
* Explore, summarize and visualize the data set - helps provide a sanity check that the data was imported properly
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


<!-- This is the syntax for commenting/hiding text for readme/markdown -->
[//]: # (Image References)

[image1]: ./examples/DannyBasicVisualization.PNG "Basic Visualization (Plot Random Image and give Shape"
[image4]: ./examples/dannyswebimages.png "Five Traffic Sign Signs"

[image2]: ./examples/DannyNormalize.PNG "Normalizing from 0->255 pixel range to 0.01->0.99"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### In this section I will describe how I addressed each rubric point [rubric points](https://review.udacity.com/#!/rubrics/481/view) in my implementation.  

---

### Data Set Summary & Exploration

#### 1. I used built in numpy functions to help summarize aspects of the data set.

* The sizes of training, test, and validation sets were as follows:  **34,799; 12,630; 4,410**.  These were obtained through applying the `.shape` of the data imported from the files.
* The shape of a traffic sign image is **32x32** and again this was obtained by using `.shape` with the specific code being:
```# Determine shape of an traffic sign image
image_shape1 = (train['features'].shape[1], train['features'].shape[2])
```

* The number of unique classes/labels in the data set is **43** and this matches the expected results from looking at the reference .csv file titled signnames.csv.  This was obtained through using `np.unique` and then applying the built in `len()` function.

#### 2. Include an exploratory visualization of the dataset.

Here is a basic exploratory visualization of the data set - It just includes plotting an image chosen from randomly from 1 of 34,799 possible images in the set.  In addition to plotting the image I also display the associated "ground truth" label associated with this image as well as printing out its shape.  This cell can be run several times so that you can sanity check each result to make sure the label matches what you are seeing -- by crossreferencing the file provided.  [*signnames.csv*](https://github.com/dannybynum/DWB-T1-P3/blob/master/signnames.csv)

![alt text][image1]

### Design and Test a Model Architecture

#### Model Step1: Pre-Processing The Images Before Feeding Them Into The CNN

<--!1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)-->

I completed two pre-processing steps for all of the datasets.  This pre-processing included (a) Conversion from color to grayscale using the *cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)* function As a first step, and (b) Normalizing the pixel values within each image (moving from possible range of 0->255 to range of 0.01->0.99).

Custom normalization function defined as follows:
```python
def normalize_grayscale(image_data):
    a = 0.01
    b = 0.99
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data)*(b - a) )/( grayscale_max - grayscale_min ) )
```

Then for each set (train, valid, test) the following was performed (example shown for train set):
```python
i=0
for i in range(n_train):
    X_train_gray[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY).reshape(32,32,-1)
    X_train_gray[i] = normalize_grayscale(X_train_gray[i])
```


After grayscale and normalizing the data the pixel values will be changed to the new "normalized" range.  I am showing this below with a simply print out to the command window before and after applying the normalization function shown above:

![alt text][image2]
The difference between the original data set and the augmented data set is just the grayscale and the normalization - no additional pre-preprocessing was needed to achieve the desired model performance. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Here are five German traffic signs that I found on the web:
![alt text][image4]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



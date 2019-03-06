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

_Basic Setup Steps Before Coding in the Notebook_
* Fork the notebook from Udacity site
	* Log into Git and then navigate to the page of interest and click on the fork button.
	* Navigate to your Git page and click on the Clone/Download button and save it locally

* Activate Environment and Notebook via Anaconda prompt

Basic commands entered are as follows:
```
cd ~\Documents\Udacity\Term1>cd DWB-T1-P3
conda env list
activate carnd-term1
Jupyter Notebook
```


<!-- This is the syntax for commenting/hiding text for readme/markdown -->
[//]: # (Image References)

[image1]: ./examples/DannyBasicVisualization.PNG "Basic Visualization (Plot Random Image and give Shape"
[image4]: ./examples/dannyswebimages.png "Five Traffic Sign Signs"

[image2]: ./examples/DannyNormalize.PNG "Normalizing from 0->255 pixel range to 0.01->0.99"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./examples/ArchCompare.PNG "Project Architecutre Comparison to LeNet Lab in Course"
[image6]: ./examples/TrainEvalCompare.PNG "Project Training Eval Compared to LeNet Lab in Course"
[image7]: ./examples/TrainingCompare.PNG "Project Training Code (Running Data Through for # of Ephocs) Compared to LeNet Lab in Course"
[image8]: ./examples/accuracy.png "Traffic Sign 5"

## Rubric Points
### In this section I will describe how I addressed each rubric point [rubric points](https://review.udacity.com/#!/rubrics/481/view) in my implementation.  

---

### Data Set Summary & Exploration

#### 1. I used built-in numpy functions to help summarize aspects of the data set.

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

<!-- 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.) -->

I completed two pre-processing steps for all of the datasets.  This pre-processing included (a) Conversion from color to grayscale using the *cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)* function As a first step, and (b) Normalizing the pixel values within each image (moving from possible range of 0->255 to range of 0.01->0.99).

The normalization step is general practice (listed as required in project instructions) to help with numerical stability -- there is a really good overview of this in the lesson 12: Tensor Flow - Section 25. Normalized Inputs and Initial Weights.

The grayscale is not required but I choose to do this based on the fact that I don't think the color of the sign has a significant impact on being able to recognize what is on the sign - therefore I believe the model should work better with the grayscale applied.  Lesson 14: Convolutional Neural Networks - Section 2. Color --- talks specifically about this concept.

Custom normalization function defined as follows:
(note that I named this normalize function "normalize_grayscale," but would have worked as-is for 3-channel color image as well)
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
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution_1 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| RELU Activation of convolutional layer output	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution_2 5x5	    | 1x1 stride, valid padding, outupts 10x10x16   |
| RELU					| RELU Activation of convolutional layer output	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten Operation		| 5x5x16 = 400, output is 400xNone				|
| Fully_Connected_1		| Input = 400, output = 120       				|
| RELU					| RELU Activation of Fully_Connected_1			|
| Fully_Connected_2		| Input = 120, output = 84       				|
| RELU					| RELU Activation of Fully_Connected_2			|
| Fully_Connected_3		| Input = 84, output = 43       				|

The model architecture is very similar to the one used in the course material for the LeNet Lab.  Below is a comparison of the code used for the lab vs that used for the project:
![alt text][image5]
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
Summary of Training steps (Training Pipeline):
* Setup "logits" equal to running the CNN (LeNet(x) function) on an input
* Setup computation of cross-entropy using a built in TensorFlow function
* Setup a loss operation using a built in TensorFlow function
* Setup an Optimizer using built in TensorFlow function "AdamOptimizer" and set this to minimize
* Start a TensorFlow Session and run batches of 128 images through the CNN for each Epoch and run the minimize training opertion previously defined


I mostly stuck with the same training approach (Optimizer etc) as was used in the LeNet lab material in the course - below are some comparisons.  The major factors that were changed were the hyperparameters and number of ephocs as follows:
Ephocs:                   5    -> 50
learning rate:            0.001-> 0.00068
sigma (initial weights):  0.1  -> 0.148



Here is a comparision of my project code with that of the LeNet Lab code in the course for the Training Pipeline and the Training Execution code:
![alt text][image6]

![alt text][image7]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* test set accuracy of 92.4%
* validation set accuracy of 94.6% 

I used an iterative approach to tweak parameters in order to achieve the required accuracy on the validation set (94.6% achieved, 93% required)
* The Initial architecture chosen was the LeNet architecture from the lab - I performed grayscale pre-processing on the images because I didn't think color was a significant factor in my being able to recognize the traffic signs so I wanted the model to be indifferent to color.
* The following graph shows iterative tweaks of the hyperparameters and how this affected accuracy - adjustments were made after noticing a good or bad impact on the accuracy.  These accuracy numbers are on the "test" dataset and the "validation" set was used at the very end with the intention of not wanting the validation results to "bleed in" to the model parameters as was described in the course material.

![alt text][image8]

For the trails the following ranges of parameters were used with adjustments being made for maximimum accuracy
The Learning Rate was changed in the following range 0.00068 to 0.0007
The Number of Ephocs was adjusted from 25 to 50
The Mu (Initial Weights Variance) was adjusted from 0.148 to 0.165
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Here are five German traffic signs that I found on the web:
![alt text][image4]

Possible Difficulties/Challenges with these images
* Image 1 doesn't have any challenges - straight on image (no tilt or rotation), and the background is also uniform/solid
* Images 2 and 5 might present a challenge because part of another sign is included on the same pole with the speed limit sign and they are only partially cropped out
* Image 3 may present a problem because it is a different representation of a 30kph sign
* Image 4 may present a little difficulty because the image perspective is not straight on, but I suspect this is also well represented in the training dataset


#### 2. Model's predictions on the new traffic signs (downloaded from web) and discussion of results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 80 kph	      		| Correct: 80 kph								| 
| 70 kph     			| Correct: 70 kph								|
| 30 kph (new type)		| Wrong: Keep Straight							|
| 30 kph	      		| Wrong: 50 kph					 				|
| 60 kph				| Correct: 60 kph      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This low accuracy isn't too much of a surprise since there were only 5 images and 1 of them (20%) was a different type of sign that looked "similar" to the typical 30kph sign.  Also as expected the model did give lower probabilities for the images that were classified incorrectly which indicates that it knew it may not be a good classification.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on the new images is shown in the following snipets:

__Image Preparation (need 32x32 and grayscale)__
```python
os.chdir('C:/Users/Bynum/Documents/Udacity/Term1/DWB-T1-P3/Signs_From_Web')
fname1 = 'DBFromWeb1.jpg'
img1 = cv2.imread(fname1)
img_res1 = cv2.resize(img_crop1,(32,32),interpolation = cv2.INTER_AREA)
img_crop1 = img1[85:384,0:299,:]     #Manually crop to a square containing only the sign
x_1 = cv2.cvtColor(img_res1, cv2.COLOR_BGR2GRAY).reshape(32,32,-1)  #fixme just trying this
x_1 = normalize_grayscale(x_1)
x_1 = x_1.astype(np.float32).reshape(-1,32,32,1) #had to convert to float32 due to error message

```

__Set up variables__:
```python
#Created and initialized with the other TensorFlow placeholders and variables
web_x = tf.placeholder(tf.float32, (None, 32, 32, 1))
logits2 = LeNet(web_x)
```

__Start a TensorFlow Session and feed in the new images and print the predection (cross-refernce to signnames.csv file manually)__
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
 
    #using np.argmax here (similar to tf.argmax) Returns the indices of the maximum values along an axis.
    Output1 = sess.run(logits2,feed_dict={web_x : x_1})
    Output_1 = np.argmax(Output1,1)
    print("Output1= ", Output_1)
```


__Start a TensorFlow Session and feed in new images to print the softmax probabilities and associated predictions__
```python
Top_5 = tf.nn.top_k(logits2,k=5)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # DWB: Compute and return softmax(x)
    return np.exp(x)/ np.sum(np.exp(x), axis=1) #changed axis to 1 here

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    #will still have to use the ".values" at the end to get an array with the actual values that can feed into softmax function
    Top_5_1 = sess.run(Top_5,feed_dict={web_x : x_1})

    print("Web_Image1:", "Ground Truth Label is: ", Web_Image_Labels[0])
    print("Top Model Prediction Labels:  ")
    print(Top_5_1.indices)
    #print("Associated Softmax Probabilities For These Predictions: ", np.around(softmax(Top_5_1.values), decimals=2))
    print("Associated Softmax Probabilities For These Predictions: ") 
    print(softmax(Top_5_1.values))
    print("\r")

```

For the five images downloaed from the web the top 5 model predictions and associated softmax probabilities is shown:

__Web_Image1: Ground Truth Label is:  5 (sign: 80kph)__
_Top Model Prediction Labels:_  
[[5 2 3 1 7]]
_Associated Softmax Probabilities For These Predictions:_ 
[[  9.90571260e-01   8.22467171e-03   6.60758466e-04   5.41050918e-04
    2.27484293e-06]]

__Web_Image1: Ground Truth Label is:  4  (sign: 70kph)__
_Top Model Prediction Labels:_  
[[4 2 1 0 5]]
_Associated Softmax Probabilities For These Predictions:_ 
[[  9.39993560e-01   4.07896936e-02   1.90166067e-02   1.98047797e-04
    2.13774911e-06]]

__Web_Image1: Ground Truth Label is:  1 (sign: 30kph)__
_Top Model Prediction Labels:_  
[[38 36 25 40 12]]
_Associated Softmax Probabilities For These Predictions:_ 
[[ 0.7860496   0.14425866  0.03702407  0.01951063  0.01315706]]

__Web_Image1: Ground Truth Label is:  1  (sign: 30kph)__
_Top Model Prediction Labels:_  
[[ 2 31 40  1  5]]
_Associated Softmax Probabilities For These Predictions:_ 
[[ 0.80455142  0.10518255  0.05427733  0.0247219   0.01126684]]

__Web_Image1: Ground Truth Label is:  3  (sign: 60kph)__
_Top Model Prediction Labels:_  
[[ 3  1  7  2 11]]
_Associated Softmax Probabilities For These Predictions:_ 
[[ 0.93508416  0.02525585  0.0232052   0.0087737   0.00768103]]



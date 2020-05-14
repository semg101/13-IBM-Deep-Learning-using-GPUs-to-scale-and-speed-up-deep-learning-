#Digit Classification using Convolution Neural Networks on CPU (External resource) 
#Import the MNIST dataset using TensorFlow built-in feature-----------------------------------------------------------------------
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Creating an interactive section-----------------------------------------------------------------
'''
You have two basic options when using TensorFlow to run your code:

    [Build graphs and run session] Do all the set-up and THEN execute a session to evaluate tensors and run operations (ops).
    [Interactive session] create your coding and run on the fly.
'''
#For this first part, we will use the interactive session that is more suitable for environments like Jupyter notebooks.
sess = tf.InteractiveSession()

#Creating placeholders-----------------------------------------------------------------------------
'''
It's a best practice to create placeholders before variable assignments when using TensorFlow. Here we'll create placeholders for inputs ("Xs") and outputs ("Ys").

Placeholder 'X': represents the "space" allocated input or the images.

    Each input has 784 pixels distributed by a 28 width x 28 height matrix.
    The 'shape' argument defines the tensor size by its dimensions.
    1st dimension = None. Indicates that the batch size, can be of any size.
    2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image.

Placeholder 'Y': represents the final output or the labels.

    10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    The 'shape' argument defines the tensor size by its dimensions.
    1st dimension = None. Indicates that the batch size, can be of any size.
    2nd dimension = 10. Indicates the number of targets/outcomes.

dtype for both placeholders: if you not sure, use tf.float32. The limitation here is that the later presented softmax function only accepts float32 or float64 dtypes. For more dtypes, check TensorFlow's documentation here.
'''
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#CNN architecture-------------------------------------------------------------------------------------------
'''
In the first part, we learned how to use a simple CNN to classify MNIST. Now we are going to expand our knowledge using a Deep Neural Network.

Architecture of our network is:

    (Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
    (Convolutional layer 1) -> [batch_size, 28, 28, 32]
    (ReLU 1) -> [?, 28, 28, 32]
    (Max pooling 1) -> [?, 14, 14, 32]
    (Convolutional layer 2) -> [?, 14, 14, 64]
    (ReLU 2) -> [?, 14, 14, 64]
    (Max pooling 2) -> [?, 7, 7, 64]
    [fully connected layer 3] -> [1x1024]
    [ReLU 3] -> [1x1024]
    [Drop out] -> [1x1024]
    [fully connected layer 4] -> [1x10]

The next cells will explore this new architecture.
'''
#Initial parameters---------------------------------------------------------------------------------------
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

#Input and output--------------------------------------------------------------------------------------
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Converting images of the data set to tensors-----------------------------------------------------
'''
The input image is a 28 pixels by 28 pixels, 1 channel (grayscale). 
In this case, the first dimension is the batch number of the image, and can be of any size (so we set it to -1). 
The second and third dimensions are width and hight, and the last one is the image channels.
'''
x_image = tf.reshape(x, [-1,28,28,1])  
x_image

#Convolutional Layer 1---------------------------------------------------------
'''
Defining kernel weight and bias

We define a kernel here. The Size of the filter/kernel is 5x5; Input channels is 1 (grayscale); 
and we need 32 different feature maps (here, 32 feature maps means 32 different filters are applied on each image. 
So, the output of convolution layer would be 28x28x32). 
In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels].
'''
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#Apply the ReLU activation Function--------------------------------------------------------------------
'''
In this step, we just go through all outputs convolution layer, convolve1, and wherever a negative number occurs,we swap it out for a 0. 
It is called ReLU activation Function.

Let f(x) is a ReLU activation function 𝑓(𝑥)=𝑚𝑎𝑥(0,𝑥).
'''
h_conv1 = tf.nn.relu(convolve1)

#Apply the max pooling--------------------------------------------------------------------------
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv1

#Convolutional Layer 2-----------------------------------------------------------------------
'''
Weights and Biases of kernels

We apply the convolution again in this layer. Lets look at the second layer kernel:

    Filter/kernel: 5x5 (25 pixels).
    Input channels: 32 (from the 1st Conv layer, we had 32 feature maps).
    64 output feature maps.

Notice: here, the input image is [14x14x32], the filter is [5x5x32], we use 64 filters of size [5x5x32], 
and the output of the convolutional layer would be 64 convolved image, [14x14x64].

Notice: the convolution result of applying a filter of size [5x5x32] on image of size [14x14x32] is an image of size [14x14x1], 
that is, the convolution is functioning on volume.
'''
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

#Convolve image with weight tensor and add biases.
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

#Apply the ReLU activation Function----------------------------------------------------------------------
h_conv2 = tf.nn.relu(convolve2)

#Apply the max pooling-------------------------------------------------------------------
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2

#Fully Connected Layer----------------------------------------------------------------
#Flattening Second Layer
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

#Weights and Biases between layer 2 and 3
#Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64); 1027 outputs to Softmax layer.
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

#Matrix Multiplication (applying weights and biases)
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

#Apply the ReLU activation Function
h_fc1 = tf.nn.relu(fcl)
h_fc1

#Dropout Layer, Optional phase for reducing overfitting---------------------------------------------
'''
It is a phase where the network "forget" some features. 
At each training step in a mini-batch, some units get switched off randomly so that it will not interact with the network. 
That is, it weights cannot be updated, nor affect the learning of the other network nodes. 
This can be very useful for very large neural networks to prevent overfitting.
'''
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

#Readout Layer (Softmax Layer)---------------------------------------------------------------
'''
Type: Softmax, Fully Connected Layer.
Weights and Biases

In last layer, CNN takes the high-level filtered images and translate them into votes using softmax. 
Input channels: 1024 (neurons from the 3rd Layer); 10 output features.
'''
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

#Matrix Multiplication (applying weights and biases)
fc=tf.matmul(layer_drop, W_fc2) + b_fc2

#Apply the Softmax activation Function
#softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probabilities.
y_CNN= tf.nn.softmax(fc)
y_CNN

#Summary of the Deep Convolutional Neural Network-------------------------------------------------------------------
'''
Now is time to remember the structure of our network

    Input - MNIST dataset
    Convolutional and Max-Pooling
    Convolutional and Max-Pooling
    Fully Connected Layer
    Processing - Dropout
    Readout layer - Fully Connected
    Outputs - Classified digits
'''

#Define functions and train the model--------------------------------------------------------------------
#Define the loss function----
'''
We need to compare our output, layer4 tensor, with ground truth for all mini_batch. 
we can use cross entropy to see how bad our CNN is working - to measure the error at a softmax layer.

The following code shows an toy sample of cross-entropy for a mini-batch of size 2 which its items have been classified. 
You can run it (first change the cell type to code in the toolbar) to see how cross entropy changes.
'''
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

#reduce_sum computes the sum of elements of (y_ * tf.log(layer4) across second dimension of the tensor, 
#and reduce_mean computes the mean of all elements in the tensor.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

#Define the optimizer----
'''
It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. 
To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) and apply gradients to variables. 
It will be done by an optimizer: GradientDescent or Adagrad.
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Define prediction---
#Do you want to know how many of the cases in a mini-batch has been classified correctly? lets count them.
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

#Define accuracy---
#It makes more sense to report accuracy using average of correct cases.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Run session, train--
sess.run(tf.global_variables_initializer())

for i in range(5000):
    start = time.time()
    batch = mnist.train.next_batch(512)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    end = time.time()
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("step", str(i), ", training accuracy", "{:.3f}".format(train_accuracy),"test accuracy", "{:.3f}".format(test_accuracy),", B_time=" , "{:.3f}".format(end - start) )

#Evaluate the model-------------------------------------------------------------------------------------
#Print the evaluation to the user
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#Visualization--------------------------------------------------------------------------------
#Do you want to look at all the filters?
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32,-1]))

!wget --output-document utils1.py http://deeplearning.net/tutorial/code/utils.py
import utils1
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  

#Do you want to see the output of an image passing through first convolution layer?
import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")

ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

#What about second convolution layer?
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

sess.close() #finish the session

%%javascript
// Shutdown kernel
Jupyter.notebook.session.delete()
#extracting the data from within tensorflow itself, using input_data module
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#Setting up an interactive session
sess = tf.InteractiveSession()

#Creating placeholders for the images (x) and the labels (y_)..not sure if predicted or actual labels
#x is 784 because that is the number of pixels for each image, y_ is 10 because that is the number of classes
#The None is used to say that the other dimension is not yet stated..im guessing that this is dictated by the number of images
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#Creating variables for our weights and scalars
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Running a session to initialise global variables. I think this means that we're essentially 'activating' the variables, i.e all those zeros
sess.run(tf.global_variables_initializer())

#creating a variable where x is multiplied by the weight and scalar added
y = tf.matmul(x, W) + b

#Takes the averaged sum of the softmax over ever image
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#So it seems that we have created an empty framework, ready to take in data, run forward
#prop on a single network, then compute the loss.
#We now move onto running back prop using a specific method called GradientDescentOptimizer
#As I understand it, this computes the gradients, parameter update step, and applies these updates to the parameters
#0.5 is the step number
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Here i believe that we are carrying out train_step 1,000 times.
#We put in 100 images and use feed_dict to replace the placeholder tensors with 
#x (image) abd y_(true label). I'm still unsure why x batch is 0 and y batch is 1
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#So far we have set up our tensors, our forward prop, back prop, and I'm 98% sure we've 
#Initiated this.
#Now we look at evaluating the model
#We look at where we predicted the correct label. tf.argmax gives the 'index of the highest entry in a tensor along some axis'
#e.g. tf.argmax(y, 1) the label our model thinks is most likely for each input.
#tf.argmax(y_, 1) is the true label. We use tf.equal to see if the two are the same
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))

#We change booleans to floats (a correct labelling -> 1) and then take the mean. I assume we want the mean to be high
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Here we can now evaluate the model's accuracy. Supposedly it will be around 92%..let's see
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Yup! 92.13%
#Nice!
#Things I'd like to be able to do now:
# upload my own handwritten digit image and see if it works...
#Provide the model an image set of fingers that show numbers
#Increase the complexity by having a range of animal pictures for example, of five different classes.
#This was a good tutorial, I can go over it a few times but really I don't yet have a confident grasp. Anyway, next is CNNs
========================================
#Upgrading to a complex CNN
#Weight initialisation - with a small amount of noise for assymetry, prevent zero
#gradients, and make slightly positive to avoid 'dead neurons' (We're usign ReLU).
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Slightly unsure here. I'm setting up some parameters for convolution and pooling
#Convolutions uses a stride of one and zero padded (moves over the pixels one at a time and no zeros added to top or obttom of vector)
#Pooling is max pooling over 2x2 blocks..whatever that means!
#Creating functions of these now to keep the code nice and clean
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Setting up the first convolutional layer
#The convolution will compute 32 features for each 5x5 patch. First two Ds are patch size
#next is number of input channels, and last is number of output channels.
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#To apply the layer we must reshape the image to a 4d tensor, 2nd and 3rd D is width and height
#Final dimension is number of colour channels
x_image = tf.reshape(x, [-1,28,28,1])

#Now we convolve the image with W and b, apply ReLU, max pool. 
#The max_pool_2x2 will reduce image size to 14x14
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#We now create a second convolutional layer. The second layer will have 64 features for every 5x5 area of the image
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Now applying something called a densely connected layer
#Image gas been reduced to 7x7, we now add a fully-connected layer with 1024 neurons to process the entire image
#We reshape the tensor from pooling layer into batch vector, xW +b, and apply ReLU
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Applying dropout to our CNN to avoid overfitting. We create a placeholder for the probability
#Of a neuron being 'dropped'. Doing it this way allows us to turn it on for training and off for testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Lastly we add a final layer, the readout layer, which will seemingly provide us with the final output
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Now that we've set up our model we want to see how good our model can be. To train and evaluate it we will use Softmax like earlier
# Some differences between now and when we earlier just used one layer are: We use ADAM instead of gradient descent
#Include an extra parameter to dictate dropout (keep_prob in feed_dict)
#We will log the results every 100th iteration (we will run this a total of 2,000 times)
#We also user tf.Session instead of tf.InteractiveSession. Not entirley sure why.
#Keep 'Cross entropy' in your mind. This seems to be the general way to measure accuracy.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
           
        ....
        step 5000, training accuracy 0.98
step 5100, training accuracy 1

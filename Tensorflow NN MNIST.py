
# coding: utf-8

# In[4]:


#extracting the data from within tensorflow itself, using input_data module
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


# In[5]:


#Setting up an interactive session
sess = tf.InteractiveSession()


# In[6]:


#Creating placeholders for the images (x) and the labels (y_)..not sure if predicted or actual labels
#x is 784 because that is the number of pixels for each image, y_ is 10 because that is the number of classes
#The None is used to say that the other dimension is not yet stated..im guessing that this is dictated by the number of images
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[7]:


#Creating variables for our weights and scalars
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[8]:


#Running a session to initialise global variables. I think this means that we're essentially 'activating' the variables, i.e all those zeros
sess.run(tf.global_variables_initializer())


# In[9]:


#creating a variable where x is multiplied by the weight and scalar added
y = tf.matmul(x, W) + b


# In[11]:


#Takes the averaged sum of the softmax over ever image
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# In[12]:


#So it seems that we have created an empty framework, ready to take in data, run forward
#prop on a single network, then compute the loss.
#We now move onto running back prop using a specific method called GradientDescentOptimizer
#As I understand it, this computes the gradients, parameter update step, and applies these updates to the parameters
#0.5 is the step number
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[14]:


#Here i believe that we are carrying out train_step 1,000 times.
#We put in 100 images and use feed_dict to replace the placeholder tensors with 
#x (image) abd y_(true label). I'm still unsure why x batch is 0 and y batch is 1
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# In[15]:


#So far we have set up our tensors, our forward prop, back prop, and I'm 98% sure we've 
#Initiated this.
#Now we look at evaluating the model
#We look at where we predicted the correct label. tf.argmax gives the 'index of the highest entry in a tensor along some axis'
#e.g. tf.argmax(y, 1) the label our model thinks is most likely for each input.
#tf.argmax(y_, 1) is the true label. We use tf.equal to see if the two are the same
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))


# In[16]:


#We change booleans to floats (a correct labelling -> 1) and then take the mean. I assume we want the mean to be high
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Here we can now evaluate the model's accuracy. Supposedly it will be around 92%..let's see
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:


#Yup! 92.13%
#Nice!
#Things I'd like to be able to do now:
# upload my own handwritten digit image and see if it works...
#Provide the model an image set of fingers that show numbers
#Increase the complexity by having a range of animal pictures for example, of five different classes.
#This was a good tutorial, I can go over it a few times but really I don't yet have a confident grasp. Anyway, next is CNNs


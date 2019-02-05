import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import math as m
import numpy as np
import random
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image = input_data.read_data_sets("MNIST_data_images", one_hot=True)
with tf.name_scope("Input"):
    input = tf.placeholder(tf.float32, [None,784],name="input")
    input_image = tf.reshape(input, [-1,28,28,1],name="input_reshaped")

with tf.name_scope("Labels"):    
    y = tf.placeholder(tf.float32, [None, 10],name="labels")

n_epoch = 50
batch_size = 800
learning_rate = 0.001

def fully_connected_layer(input,size_in,size_out):
    W = tf.Variable(tf.truncated_normal([size_in,size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[1,size_out]), name="b")
    output = tf.matmul(input, W)
    output = tf.add(output, b)
    return output
    
def conv2d(input, W):
    return tf.nn.conv2d(input, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def biases(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


keep_prob = tf.placeholder(tf.float32,name="keep_prob")

with tf.name_scope("Conv1"):
    W_conv1 = weight([8, 8, 1, 50])
    b_conv1 = biases([50])
    h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Conv2"):
    W_conv2 = weight([5, 5, 50, 64])
    b_conv2 = biases([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("Conv3"):
    W_conv3 = weight([3, 3, 64, 64])
    b_conv3 = biases([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    
with tf.name_scope("Conv4"):
    W_conv4 = weight([3, 3, 64, 32])
    b_conv4 = biases([32])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    
with tf.name_scope("Dense1"):
    W_fc2 = weight([7 * 7 * 32, 1024])
    b_fc2 = biases([1024])
    h_pool2_flat = tf.reshape(h_conv4, [-1, 7*7*32])
    h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
with tf.name_scope("Dense2"):
    W_fc3 = weight([1024, 10])
    b_fc3 = biases([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    
    
with tf.name_scope("Training"):    
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("Test"):
    with tf.name_scope("Correctes"):
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))       
    acc_scalar_train = tf.summary.scalar('accuracy_train', accuracy)     
    acc_scalar_valid = tf.summary.scalar('accuracy_valid', accuracy)     
    acc_scalar_rot_train = tf.summary.scalar('accuracy_rot_train', accuracy)
    acc_scalar_rot_valid = tf.summary.scalar('accuracy_rot_valid', accuracy)
    
with tf.name_scope("Rotation"):
    image_t = tf.placeholder(tf.float32, [None,784], name="img_input")
    angle = tf.placeholder(tf.float32, [None,], name="angle")
    image_t_reshaped = tf.reshape(image_t, [28,28], name="img_reshap")
    image_t_rot = tf.contrib.image.rotate(image_t,angle[1],interpolation='BILINEAR',name="rotate")
    image_t_rot_reshaped = tf.reshape(image_t_rot, [28,28])
    image_t_rot_flattened = tf.reshape(image_t_rot, [-1,784])

rot_images = image.train.images
rot_labels = image.train.labels
rot_set = image.train.images
rot_set_labels = image.train.labels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def rotate_tensormod(rot_set,rot_set_labels):    
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("./logs", graph=sess.graph)
        print("\n\n\n\n\n Début préprocess")
        sample_size = 54000
        sample = random.sample(range(1, 55000), sample_size) 
        image_dict = []
        angle_dict = []
        n=0
        for i in sample:
            n += 1
            if n%500==0:
                print("Rotation",n,"out of",sample_size)
            angle_dict.append(random.uniform(-m.pi/5, m.pi/5))
            image_dict.append(image.train.images[i])
            rot_set_labels = np.concatenate((rot_set_labels, [image.train.labels[i]]))
        img = sess.run(image_t_rot_flattened, feed_dict={image_t:image_dict, angle:angle_dict})        
        rot_set = np.concatenate((rot_set, img))
        print("Préprocess effectué \n\n\n\n\n")  
        writer.close()
    return rot_set, rot_set_labels     



def visualize_rotation():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image_t = image.train.images[i]
        image_t = image_t.reshape([28,28])
        image_t_rot = tf.contrib.image.rotate(image_t,m.pi/12,interpolation='BILINEAR')
        image_t_rot = tf.reshape(image_t_rot, [28,28])
        image_t_rot = tf.reshape(image_t_rot, [-1,784])                  
        fig = plt.figure()    
        a = fig.add_subplot(1,2,1)
        img_rot = sess.run(image_t_rot)
        img_plot_1 = plt.imshow(img_rot, cmap='gray_r')
        img = mnist.train.images[0]
        img = img.reshape([28,28])
        b = fig.add_subplot(1,2,2)
        img_plot = plt.imshow(img, cmap='gray_r')
        print(img==image_t_rot)
        plt.show()
       
       
def train_rot():
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs", graph=sess.graph)
        print("\n\n\n\n\nTaille train set:",len(rot_images),"\n")    
        
        feed_valid = {input:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0}
        feed_train = {input:rot_images[0:10000], y:rot_labels[0:10000], keep_prob: 1.0}        
        
        for epoch in range(n_epoch):
            for i in range(int(len(rot_images)/batch_size)):
                batch_x = rot_images[i*batch_size:(i+1)*batch_size] 
                batch_y = rot_labels[i*batch_size:(i+1)*batch_size]
                train_dict = {input:batch_x, y:batch_y, keep_prob: 0.5}
                sess.run(optimizer, feed_dict=train_dict)        
            
            sum_acc_rot_train = sess.run(acc_scalar_rot_train, feed_dict=feed_train)
            sum_acc_rot_valid = sess.run(acc_scalar_rot_valid, feed_dict=feed_valid)   
            
            accuracy_validation = sess.run(accuracy, feed_dict=feed_valid)   
            
            writer.add_summary(sum_acc_rot_train, global_step=epoch)
            writer.add_summary(sum_acc_rot_valid, global_step=epoch)
            
            print("\naccuracy_test:",accuracy_validation*100,"|| epoch:", epoch+1)            
        writer.close()

def train_norm(learning_rate):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs", graph=sess.graph)
        print("\n\n\n\n\nTaille train set:",len(rot_images),"\n") 
      
        feed_valid = {input:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0}
        feed_train = {input:rot_images[0:10000], y:rot_labels[0:10000], keep_prob: 1.0}
        
        for epoch in range(n_epoch):
            for i in range(int(len(rot_images)/batch_size)):               
                batch_x = rot_images[i*batch_size:(i+1)*batch_size] 
                batch_y = rot_labels[i*batch_size:(i+1)*batch_size]
                train_dict = {input:batch_x, y:batch_y, keep_prob: 0.5}
                o = sess.run(optimizer, feed_dict=train_dict)        
                

            
            sum_acc_train = sess.run(acc_scalar_train, feed_dict=feed_train)
            sum_acc_valid = sess.run(acc_scalar_valid, feed_dict=feed_valid)   
            
            accuracy_validation = sess.run(accuracy, feed_dict=feed_valid)   
            
            writer.add_summary(sum_acc_train, global_step=epoch)
            writer.add_summary(sum_acc_valid, global_step=epoch)
            
            print("\naccuracy_validation:",accuracy_validation*100,"|| epoch:", epoch+1)            
        writer.close()


train_norm(learning_rate)
# rot_images, rot_labels = rotate_tensormod(rot_set,rot_set_labels)
# train_rot()








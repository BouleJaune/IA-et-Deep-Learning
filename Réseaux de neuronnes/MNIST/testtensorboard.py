import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
W=tf.Variable(tf.truncated_normal([500,500],stddev=0.1))

        
with tf.name_scope('CNN1'):
    with tf.name_scope('W'):
        mean = tf.reduce_mean(W)
        tf.summary.scalar('mean', mean)
        
sess = tf.Session()
writer = tf.summary.FileWriter('/logtest', graph=sess.graph)
sess.run(tf.global_variables_initializer())
writer.close()
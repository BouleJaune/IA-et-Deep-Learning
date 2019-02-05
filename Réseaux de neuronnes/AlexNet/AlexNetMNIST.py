import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np

batch_size = 32
input_size = 784
n_classes = 10
nb_epochs = 20 

input_r = tf.placeholder(tf.float32, [None, input_size], name="input")
input = tf.reshape(input_r, shape=[-1, 28, 28, 1], name="Input_reshaped")

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
y = tf.placeholder(tf.float32, [None,n_classes], name="Labels")

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name="Weights")

def bias(shape, valeur):
    return tf.Variable(tf.constant(valeur,shape=shape), name="Biases")
    
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1,4,4,1], padding="SAME")

def fully_connected_layer(input,W,b):
    return tf.add(tf.matmul(input,W),b)

def overlap_pooling(input):
    return tf.nn.max_pool(input, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
   

with tf.name_scope("Conv1"):
    taille_1 = 12
    with tf.name_scope("Group1"):
        filtres_conv11 = weight([11,11,1,taille_1])
        b_conv11 = bias([taille_1],0.0)
        h_conv11 = conv2d(input, filtres_conv11) + b_conv11
        h_norm11 = tf.nn.relu(tf.nn.local_response_normalization(h_conv11, bias=2, alpha=0.0001, beta=0.75))
        h_pool11 = overlap_pooling(h_conv11) 
    with tf.name_scope("Group2"):
        filtres_conv12 = weight([11,11,1,taille_1])
        b_conv12 = bias([taille_1],0.0)
        h_conv12 = conv2d(input, filtres_conv12) + b_conv12
        h_norm12 = tf.nn.relu(tf.nn.local_response_normalization(h_conv12, bias=2, alpha=0.0001, beta=0.75))
        h_pool12 = overlap_pooling(h_conv12)
   
with tf.name_scope("Conv2"):
    with tf.name_scope("Group1"):
        filtres_conv21 = weight([5,5,taille_1,12])
        b_conv21 = bias([12],1.0)
        h_conv21 = conv2d(h_pool11, filtres_conv21) + b_conv21
        h_norm21 = tf.nn.relu(tf.nn.local_response_normalization(h_conv21, bias=2, alpha=0.0001, beta=0.75))
        h_pool21 = overlap_pooling(h_conv21)
    with tf.name_scope("Group2"):    
        filtres_conv22 = weight([5,5,taille_1,12])
        b_conv22 = bias([12],1.0)
        h_conv22 = conv2d(h_pool12, filtres_conv22) + b_conv22
        h_norm22 = tf.nn.relu(tf.nn.local_response_normalization(h_conv22, bias=2, alpha=0.0001, beta=0.75))
        h_pool22 = overlap_pooling(h_conv22)
    h_pool2 = tf.concat([h_pool21,h_conv22],axis=3)

with tf.name_scope("Conv3"):
    with tf.name_scope("Group1"):    
        filtres_conv31 = weight([3,3,24,10])
        b_conv31 = bias([10],0.0)
        h_conv31 = tf.nn.relu(conv2d(h_pool2, filtres_conv31) + b_conv31)
    with tf.name_scope("Group2"):
        filtres_conv32 = weight([3,3,24,10])
        b_conv32 = bias([10],0.0)
        h_conv32 = tf.nn.relu(conv2d(h_pool2, filtres_conv32) + b_conv32)
    
with tf.name_scope("Conv4"):
    with tf.name_scope("Group1"):
        filtres_conv41 = weight([3,3,10,10])
        b_conv41 = bias([10],1.0)
        h_conv41 = tf.nn.relu(conv2d(h_conv31, filtres_conv41) + b_conv41)
    with tf.name_scope("Group2"):
        filtres_conv42 = weight([3,3,10,10])
        b_conv42 = bias([10],1.0)
        h_conv42 = tf.nn.relu(conv2d(h_conv32, filtres_conv42) + b_conv42)
    
with tf.name_scope("Conv5"):
    with tf.name_scope("Group1"):
        filtres_conv51 = weight([3,3,10,10])
        b_conv51 = bias([10],1.0)
        h_conv51 = conv2d(h_conv41, filtres_conv51) + b_conv51
        h_norm51 = tf.nn.relu(tf.nn.local_response_normalization(h_conv51, bias=2, alpha=0.0001, beta=0.75))
        h_pool51 = overlap_pooling(h_conv51)
        shape_51 = h_pool51.get_shape().as_list()
        size_51 = np.prod(shape_51[1:]) 
        h_pool51 = tf.reshape(h_pool51,[-1,size_51])
    with tf.name_scope("Group2"):
        filtres_conv52 = weight([3,3,10,10])
        b_conv52 = bias([10],1.0)
        h_conv52 = conv2d(h_conv42, filtres_conv52) + b_conv52
        h_norm52 = tf.nn.relu(tf.nn.local_response_normalization(h_conv52, bias=2, alpha=0.0001, beta=0.75))
        h_pool52 = overlap_pooling(h_conv52)
        shape_52 = h_pool51.get_shape().as_list()
        size_52 = np.prod(shape_52[1:]) 
        h_pool52 = tf.reshape(h_pool52,[-1,size_52]) #16384

with tf.name_scope("Dense1"):
    with tf.name_scope("Group1"):
        w_fc11 = weight([size_51,208])
        b_fc11 = bias([208],0.0)
        fc11 = tf.nn.relu(fully_connected_layer(h_pool51, w_fc11, b_fc11))
        fc1_drop1 = tf.nn.dropout(fc11, keep_prob)
    with tf.name_scope("Group2"):
        w_fc12 = weight([size_52,208])
        b_fc12 = bias([208],0.0)
        fc12 = tf.nn.relu(fully_connected_layer(h_pool52, w_fc12, b_fc12))
        fc1_drop2 = tf.nn.dropout(fc12, keep_prob)

with tf.name_scope("Dense2"):
    with tf.name_scope("Group1"):
        w_fc21 = weight([208,208])
        b_fc21 = bias([208],0.0)
        fc21 = tf.nn.relu(fully_connected_layer(fc1_drop1, w_fc21, b_fc21))
        fc2_drop1 = tf.nn.dropout(fc21, keep_prob)
    with tf.name_scope("Group1"):
        w_fc22 = weight([208,208])
        b_fc22 = bias([208],0.0)
        fc22 = tf.nn.relu(fully_connected_layer(fc1_drop2, w_fc22, b_fc22))
        fc2_drop2 = tf.nn.dropout(fc22, keep_prob)
    fc2 = tf.concat([fc2_drop1,fc2_drop2],axis=1)
    
with tf.name_scope("Dense3"):
    w_fc3 = weight([208*2,n_classes])
    b_fc3 = bias([n_classes],0.0)
    prediction = tf.nn.softmax(fully_connected_layer(fc2, w_fc3, b_fc3))

with tf.name_scope("Training"):    
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope("Test"):
    with tf.name_scope("Correctes"):
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    with tf.name_scope("accuracy"):
        accuracy_t = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))       

            
def train_neural_network(input): 
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())      
        writer = tf.summary.FileWriter("./logs", graph=sess.graph)
        for epoch in range(nb_epochs):                 
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                x_b, y_b = mnist.train.next_batch(batch_size)
                

                feed_train = {input_r: x_b, y: y_b, keep_prob: 0.5}
                o, c = sess.run([optimizer, cost], feed_dict=feed_train)
                # sum_biases = sess.run(biases_histo, feed_dict=feed_train)
                # sum_weights = sess.run(weights_histo, feed_dict=feed_train)
                # for i in range(len(sum_biases)):
                    # writer.add_summary(sum_biases[i], global_step=epoch)
                    # writer.add_summary(sum_weights[i], global_step=epoch)          
                epoch_loss += c         
            
            feed_test = {input_r:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0}
            # sum_acc = sess.run(acc_scalar, feed_dict=feed_test)              
            # writer.add_summary(sum_acc, global_step=epoch)            
            accuracy = accuracy_t.eval(feed_test)     
            # accuracy_vec.append(accuracy)
            print('\nAccuracy', 100*accuracy, '| Epoch', epoch,'| loss:', epoch_loss,'\n')                           
        writer.close()
        print()
        
train_neural_network(input)  


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import os
import math
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
# from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.summary.FileWriterCache.clear()
# Hyperparamètres :
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_n_h_layers = Integer(low=3, high=5, name='n_h_layers')
dim_n_per_layer = Integer(low=16, high=512, name='n_per_layer')
dimensions = [dim_learning_rate, dim_n_h_layers, dim_n_per_layer]
default_parameters = [1e-5, 3, 200 ]
loss_function = tf.nn.softmax_cross_entropy_with_logits_v2
activation_f = tf.nn.relu
patience = 10 # early stopping epoch
batch_size = 500
n_epochs = 100

tf_log = 'tf.log'

save_file = './checkpoints/model.ckpt'
path_best_model = './best_model.ckpt' 

def log_dir_name(learning_rate, n_h_layers,
                 n_per_layer):
    s = "./logs/lr_{0:.0e}_layers_{1}_neurones_{2}/"
    log_dir = s.format(learning_rate, n_h_layers, n_per_layer)
    return log_dir
    
def neural_network_model(data,learning_rate, n_h_layers,
                            n_per_layer, activation_f):
    n_neurones = [784]
    n_classes = 10                                                               
    n_neurones.extend([n_per_layer]*n_h_layers)
    n_neurones.append(n_classes)
    hidden_layers = []
    etat_neurones = [data]
    weights_histo = []
    biases_histo = []
    
    for i in range(n_h_layers-1):
        with tf.name_scope('Hidden-layer-'+str(i+1)):
            layer_tmp = {'weights':tf.Variable(tf.truncated_normal([n_neurones[i], n_neurones[i+1]] ,stddev=0.1,seed=123), name='Weights'),
                                 'biases':tf.Variable(tf.constant( 0.0 , shape=[n_neurones[i+1]]), name='Biases')}
        hidden_layers.append(layer_tmp)
        weights_histo.append(tf.summary.histogram('weights-'+str(i+1), layer_tmp['weights']))
        biases_histo.append(tf.summary.histogram('biases-'+str(i+1), layer_tmp['biases']))
        
    with tf.name_scope('layer-'+str(n_h_layers)):                         
        output_layer = {'weights':tf.Variable(tf.truncated_normal( [n_neurones[n_h_layers],n_neurones[n_h_layers+1]] ,stddev=0.1,seed=123), name='Weights'),
                        'biases':tf.Variable(tf.constant( 0.0, shape=[n_neurones[n_h_layers+1]]),name='Biases')}                                                                 
    
    for i in range(n_h_layers-1):
        with tf.name_scope('Application_layer'+str(i+1)):
            with tf.name_scope('Multiplication-'+str(i+1)):
                etat_neurones_tmp = tf.matmul(etat_neurones[i], hidden_layers[i]['weights'])
            with tf.name_scope('Ajout-biais-'+str(i+1)):
                etat_neurones_tmp = tf.add(etat_neurones_tmp, hidden_layers[i]['biases'])            
            with tf.name_scope('RectLin-'+str(i+1)):
                etat_neurones_tmp = activation_f(etat_neurones_tmp)            
        etat_neurones.append(etat_neurones_tmp)  
        tf.summary.histogram('activations-'+str(i+1), etat_neurones_tmp)
        
    with tf.name_scope('Application_layer_f'):    
        with tf.name_scope('Multiplication-finale'):
            output = tf.matmul(etat_neurones[-1], output_layer['weights'])
        with tf.name_scope('Ajout-biais-final'):
            output = tf.add(output, output_layer['biases'])   
    return output, biases_histo, weights_histo              

@use_named_args(dimensions=dimensions)
def train_neural_network(learning_rate, n_h_layers,
                            n_per_layer): 
   
    with tf.name_scope('Labels'):   
        y = tf.placeholder(tf.float32)          
    with tf.name_scope('Input-Data'):
        x_input = tf.placeholder(tf.float32,[None, 784])
        
    log_dir = log_dir_name(learning_rate, n_h_layers, n_per_layer)    
    feed_test = {x_input: mnist.test.images, y: mnist.test.labels}        
    prediction, biases_histo, weights_histo = neural_network_model(x_input,learning_rate, n_h_layers,
                            n_per_layer, activation_f)                                     
        
    with tf.name_scope('Hyperparameters'):
        with tf.name_scope('learning_rate'):
            lr = tf.Variable(tf.constant(learning_rate, shape=[]))
            lr_scal = tf.summary.scalar('learning_rate',lr)
        with tf.name_scope('n_h_layers'):
            nhl = tf.Variable(tf.constant(n_h_layers, shape=[]))
            nhl_scal = tf.summary.scalar('num_h_layers',nhl)
        with tf.name_scope('n_per_layer'):            
            npl = tf.Variable(tf.constant(n_per_layer, shape=[]))                      
            npl_scal = tf.summary.scalar('n_per_layer',npl)    
          
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cost = tf.reduce_mean(loss_function(logits=prediction,labels=y), name='cost')                
            
    with tf.name_scope('train'):       
        optimi = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            
    with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                with tf.name_scope('accuracy'):
                    red_mea = tf.reduce_mean(tf.cast(correct,tf.float32))
                    
    cost_scalar = tf.summary.scalar('cross_entropy',cost)                   
    acc_scalar = tf.summary.scalar('accuracy',red_mea)
    # merge = tf.summary.merge_all()
    saver = tf.train.Saver()      

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # try:
            # epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            # print('STARTING:',epoch)
        # except:
            # epoch = 1
        # if epoch != 1:
            # saver.restore(sess,save_file)  
        epoch=1
        accuracy_vec=[]
            
        while epoch <= n_epochs:                 
            epoch_loss = 0

            for i in range(int(mnist.train.num_examples/batch_size)):
                x_b, y_b = mnist.train.next_batch(batch_size)
                feed_train = {x_input: x_b, y: y_b}
                o, c = sess.run([optimi, cost], feed_dict=feed_train)
                sum_biases = sess.run(biases_histo, feed_dict=feed_train)
                sum_weights = sess.run(weights_histo, feed_dict=feed_train)
                for i in range(len(sum_biases)):
                    writer.add_summary(sum_biases[i], global_step=epoch)
                    writer.add_summary(sum_weights[i], global_step=epoch)          
                epoch_loss += c         
                                    
            # sum_hyperparameters = sess.run(hyperparameters)
            
            sum_acc = sess.run(acc_scalar, feed_dict=feed_test)              
            writer.add_summary(sum_acc, global_step=epoch)            
            accuracy = red_mea.eval(feed_test)     
            accuracy_vec.append(accuracy)
            print('\nAccuracy', 100*accuracy, '| Epoch', epoch,'| loss:', epoch_loss,'\n')                           
            saver.save(sess, save_file+str(epoch))
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n')               
                # if epoch > patience and all(accuracy_vec[-(i+1)] > accuracy_vec[-1] for i in range(1,patience+1)):
                    # epoch = n_epochs    
            epoch += 1           
        print(log_dir)
        writer.close()
    print()
    return -accuracy    
        
# D:\Documents\OneDrive\Documents\Cours\Machine learning\sentdex playlist\Tensorflow\MNIST           
search_result = gp_minimize(func=train_neural_network,
                            dimensions=dimensions,
                            x0=default_parameters,
                            acq_func='EI', # Expected Improvement.
                            n_calls=101)
                           
print(search_result.x)
print("""Best parameters:
- learning_rate=%.6f
- profondeur=%d
- neurons per layer=%d
""" % (search_result.x[0], search_result.x[1],  search_result.x[2]))

plot_convergence(search_result)
plt.show()  
# train_neural_network(default_parameters)    
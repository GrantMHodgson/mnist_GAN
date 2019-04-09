import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.reset_default_graph()

mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

#
# -------------------------------------------
#
# Global variables

batch_size = 128
z_dim = 10

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def lrelu( x, leak=0.2, name="lrelu" ):
    return tf.maximum( x, leak*x )

def deconv2d( in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0 ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [ k_w, k_h, output_shape[-1], in_var.get_shape()[-1] ]

    with tf.variable_scope( name ):    
        W = tf.get_variable( "W", var_shape,
                             initializer=tf.truncated_normal_initializer( stddev=0.02 ) )
        b = tf.get_variable( "b", [output_shape[-1]],
                             initializer=tf.constant_initializer( bias_val ))

        deconv = tf.nn.conv2d_transpose( in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1] )
        deconv = tf.reshape( tf.nn.bias_add( deconv, b), deconv.get_shape() )
        return deconv

def conv2d( in_var, output_dim, name="conv2d" ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv

#
# ==================================================================
# ==================================================================
# ==================================================================
#



# the generator should accept a (tensor of multiple) 'z' and return an image
# z will be [None,z_dim]

def gen_model( z ):

    # H1: A linear layer, mapping z to 128*7*7 features, followed by a relu
    g_h1 = linear(z, 6272, "g_h1")
    #quick reshape
    g_h1 = tf.reshape(g_h1, [batch_size, 7, 7, 128])
    g_h1 = tf.nn.relu(g_h1)
    
    
    # D2: a deconvolution layer, mapping H1 to a tensor that is [batch_size,14,14,128], followed by a relu
    g_d2 = deconv2d(g_h1,[batch_size,14,14,128], "g_d2")
    g_d2 = tf.nn.relu(g_d2)

    # D3: a deconvolution layer, mapping D2 to a tensor that is [batch_size,28,28,1]
    g_d3 = deconv2d(g_d2, [batch_size, 28, 28, 1], "g_d3")
    # Note D3 is reshaped to be [batch_size,784] for compatibility with the discriminator
    g_d3 = tf.reshape(g_d3, [batch_size, 784])

    # The final output should be sigmoid of D3
    g_out = tf.sigmoid(g_d3, name='g_out')
    return g_out

# -------------------------------------------
    
# the discriminator should accept a (tensor of muliple) images and
# return a probability that the image is real
# imgs will be [None,784]

def disc_model( imgs ):
    imgs = tf.reshape( imgs, [ batch_size, 28, 28, 1 ] )

    # H0: A 2d convolution on imgs with 32 filters, followed by a leaky relu
    d_h0 = conv2d(imgs, 32, "d_h0") 
    d_h0 = lrelu(d_h0)
    
    # H1: A 2d convolution on H0 with 64 filters, followed by a leaky relu
    d_h1 = conv2d(d_h0, 64, "d_h1")
    d_h1 = lrelu(d_h1)
    d_h1 = tf.reshape( d_h1, [ batch_size, -1 ] )

    # H2: A linear layer from H1 to a 1024 dimensional vector
    d_h2 = linear(d_h1, 1024, "d_h2")

    # H3: A linear layer mapping H2 to a single scalar (per image)
    d_h3 = linear(d_h2, 1, "d_h3") 

    # The final output should be a sigmoid of H3.
    d_out = tf.sigmoid(d_h3, name='d_out')

    return d_out

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# Create the computation graph, cost function, and training steps

# Placeholders should be named 'z' and 'true_images'
z_dim = 100
z = tf.placeholder(tf.float32, shape=[None, z_dim], name = 'z')
true_images = tf.placeholder(tf.float32, shape=[None, 784], name='true_images')




with tf.variable_scope("generator"):
    # We pass the z variable into the generative model, and call the output sample_images
    sample_images = gen_model(z)

with tf.variable_scope("discriminator"):
    # We pass some true images into the discriminator, and get back some probabilities.
    true_probs = disc_model(true_images)

with tf.variable_scope("discriminator", reuse=True):
    # We pass some sampled images into the discriminator, and get back some (different) probabilities.
    sample_probs = disc_model(sample_images)


    
t_vars = tf.trainable_variables()   
# We construct a loss function for the discriminator that attempts to maximize 
# the log of the output probabilities on the true images and the log of 1.0 - the output 
# probabilities on the sampled images; these two halves can be summed together
with tf.variable_scope("optim"):
    d_loss = -tf.reduce_mean(tf.add(tf.log(true_probs), tf.log(1-sample_probs)))

    d_vars = [var for var in t_vars if 'd_' in var.name]
    d_optim = tf.train.AdamOptimizer( 0.0002, beta1=0.5 ).minimize( d_loss, var_list=d_vars )


    # We construct a loss function for the generator that attempts to maximize the 
    # log of the output probabilities on the sampled images
    g_loss = -tf.reduce_mean((tf.log(sample_probs)))
    g_vars = [var for var in t_vars if 'g_' in var.name]
    g_optim =  tf.train.AdamOptimizer( 0.0002, beta1=0.5 ).minimize( g_loss, var_list=g_vars )
    
    # d_acc calculates classification accuracy on a batch. It checks the output 
    # probabilities of the discriminator on the real and sampled images, and see if they're 
    # greater (or less) than 0.5.
    d_acc_true = tf.reduce_mean(tf.cast(tf.greater(true_probs, .5), tf.float32))
    d_acc_sample = tf.reduce_mean(tf.cast(tf.greater(1-sample_probs, .5), tf.float32))
    d_acc = tf.truediv(tf.add(d_acc_true, d_acc_sample), tf.cast(2, tf.float32))
    



# The output of the generator should be named 'sample_images'
    
#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

print("itr\td_loss g_loss d_acc")

for i in range( 5000 ):
    batch = mnist.train.next_batch( batch_size )
    batch_images = batch[0]

    sampled_zs = np.random.uniform( low=-1, high=1, size=(batch_size, z_dim) ).astype( np.float32 )
    sess.run( d_optim, feed_dict={ z:sampled_zs, true_images: batch_images } )

    for j in range(3):
        sampled_zs = np.random.uniform( low=-1, high=1, size=(batch_size, z_dim) ).astype( np.float32 )
        sess.run( g_optim, feed_dict={ z:sampled_zs } )
    
    if i%10==0:
        d_acc_val,d_loss_val,g_loss_val = sess.run( [d_acc,d_loss,g_loss],
                                                    feed_dict={ z:sampled_zs, true_images: batch_images } )
        print("%d\t%.2f %.2f %.2f" % ( i, d_loss_val, g_loss_val, d_acc_val ))

summary_writer.close()

#
#  show some results
#
sampled_zs = np.random.uniform( -1, 1, size=(batch_size, z_dim) ).astype( np.float32 )
simgs = sess.run( sample_images, feed_dict={ z:sampled_zs } )
simgs = simgs[0:64,:]

tiles = []
for i in range(0,8):
    tiles.append( np.reshape( simgs[i*8:(i+1)*8,:], [28*8,28] ) )
plt.imshow( np.hstack(tiles), interpolation='nearest', cmap=matplotlib.cm.gray )
plt.colorbar()
plt.savefig('GAN_output.png')

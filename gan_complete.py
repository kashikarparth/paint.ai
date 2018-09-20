import tensorflow as tf
import batch_feeding as data
import os
import scipy.misc
import numpy as np
import tensorflow as tf

import keras.backend as K

import matplotlib.pyplot as plt

from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Flatten, Activation
from keras.layers import Input, UpSampling2D
from keras.models import Model

tf.reset_default_graph()


def make_generator(Xk_g):
    up = UpSampling2D(size=(4,4))(Xk_g)
    x = Deconvolution2D(1024,4,4,output_shape = (None,1024,8,8), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    x = Deconvolution2D(512,4,4,output_shape = (None,512,16,16), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    x = Deconvolution2D(256,4,4,output_shape = (None,256,32,32), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    x = Deconvolution2D(128,4,4,output_shape = (None,128,64,64), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    x = Deconvolution2D(64,4,4,output_shape = (None,64,128,128), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    x = Deconvolution2D(3,4,4,output_shape = (None,3,256,256), activation = None)(up)
    x = LeakyReLU(0.2)(x)
    g = x
    return g

def make_discriminator(Xk_d):
    
    
    x = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(Xk_d)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(nb_filter=126, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(nb_filter=256, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, subsample=(2,2),activation=None, border_mode='same',dim_ordering='tf')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    
    d_real = Dense(1024)(x)
    d_real = LeakyReLU(0.2)(d_real)
    d_real= Dense(1)(d_real)
      
    d_class = Dense(1024)(x)
    d_class = LeakyReLU(0.2)(d_class)
    d_class = Dense(1024)(x)
    d_class = LeakyReLU(0.2)(d_class)
    d_class = Dense(512)(d_class)
    d_class = LeakyReLU(0.2)(d_class)
    d_class = Dense(27)(d_class)
      
    return d_real,d_class

default_opt = { 'lr' : 5e-5, 'c' : 1e-2, 'n_critic' : 5 }

                   
class painter(object):
    def __init__(self,opt_params=default_opt):
        
        ##DATA RELATED#####
        
        self.root='D:/Parth Kashikar/AI/Projects/CAN/smallimages'
        self.cumulative = []
        self.path = []
        self.pictures_done = 0
        
        i = 0
        summation = 0
        
        for subdirs,dirs,files in os.walk(self.root):
            if(i>0):
                self.path.append(subdirs)
            i=i+1
            
        i = 0
        
        for subdirs,dirs,files in os.walk(self.root):
            if(i>0):
                for _,_,files in os.walk(subdirs):
                    summation = summation + len(files)
                    self.cumulative.append(summation)
            i = i + 1
            
        #########MODEL RELATED######
        self.sess = tf.Session()
        K.set_session(self.sess)
        
        self.c = opt_params['c']
        
        with tf.name_scope('generator'):
            Xk_g = Input(shape=[1,1,2048])
            g = make_generator(Xk_g)
        
        with tf.name_scope('discriminator'):
            Xk_d = Input(shape=[256,256,3])
            d,dc = make_discriminator(Xk_d)
        
        g_net = Model(input=Xk_g, output=g)
        d_net = Model(input=Xk_d, output=d)
        dc_net = Model(input=Xk_d, output=dc)
        
        
        X_g = tf.placeholder(tf.float32,shape=[None,1,1,2048],name = 'X_g')
        X_d = tf.placeholder(tf.float32,shape=[None,256,256,3], name = 'X_d')
        X_dc = tf.placeholder(tf.float32,shape=[None,27], name = 'X_dc')
        G_dc = tf.placeholder(tf.float32,shape=[None,27], name = 'G_dc')
        
        self.inputs = X_g,X_d,X_dc,G_dc

        
        self.w_g = [w for w in tf.global_variables() if 'generator' in w.name]
        self.w_d = [w for w in tf.global_variables() if 'discriminator' in w.name]
    
        d_real = d_net(X_d)
        d_real_class = dc_net(X_d)
        d_fake = d_net(g_net(X_g))
        d_fake_class = dc_net(g_net((X_g)))
        self.P = g_net(X_g)
        
        self.loss_g = tf.reduce_mean(d_fake) - tf.norm(tf.subtract(d_fake_class,G_dc))
        self.loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        self.loss_dc = - tf.norm(tf.subtract(d_real_class,X_dc))
        
        self.d_real = tf.reduce_mean(d_real)
        self.d_fake = tf.reduce_mean(d_fake)
        self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
        self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

        lr = opt_params['lr']
        optimizer_g = tf.train.RMSPropOptimizer(lr)
        optimizer_d = tf.train.RMSPropOptimizer(lr)
        optimizer_dc = tf.train.RMSPropOptimizer(lr)
        
        gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
        gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)
        gv_dc = optimizer_dc.compute_gradients(self.loss_dc, self.w_d)
        
        self.train_op_g = optimizer_g.apply_gradients(gv_g)
        self.train_op_d = optimizer_d.apply_gradients(gv_d)
        self.train_op_dc = optimizer_dc.apply_gradients(gv_dc)
        
        self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.w_d]
        
    def next_batch(self, train = True):
        
        images = []
        for i in range(len(self.cumulative)):
            if(self.cumulative[i]>self.pictures_done):
                current_index = i
                if(self.cumulative[current_index]-self.pictures_done >= 25):
                    next_batch_size = 25
                else:
                    next_batch_size = self.cumulative[current_index]-self.pictures_done
                break
        
        if(current_index>0):
            skip_images = self.pictures_done - self.cumulative[current_index-1]
        else:
            skip_images = self.pictures_done
        
        for _,_,files in os.walk(self.path[current_index]):
            for i in range(len(files)):
                if(i>=skip_images and i< skip_images + next_batch_size):
                    image = scipy.misc.imread(self.path[current_index] + '/' + files[i])
                    images.append(image)
        
        images = np.asarray(images, dtype = np.float32)
        print(np.shape(images))
        D_c = np.zeros(shape = [next_batch_size,27])
        D_c[:,current_index] = 1
        G_c = 0.8*D_c
        noise = np.random.randn(next_batch_size,1,1,2048)
        noise[:,0,0,0] = current_index
        
        X_g,X_d,X_dc,G_dc = self.inputs
        
        self.pictures_done = self.pictures_done + next_batch_size
        
        return {X_g : noise, X_d : images, X_dc : D_c, G_dc : G_c,  K.learning_phase() : train}
            
    def train_g(self, feed_dict):
        _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict=feed_dict)
        return loss_g

    def train_d(self, feed_dict):
        self.sess.run(self.clip_updates, feed_dict=feed_dict)
        self.sess.run(self.train_op_d, feed_dict=feed_dict)
        self.sess.run(self.train_op_dc,feed_dict=feed_dict)
        return self.sess.run(self.loss_d, feed_dict=feed_dict)

    
    def fit(self,n_epochs = 1, logdir='gan-run'):
                
        if tf.gfile.Exists(logdir): tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        loss_d = []
        loss_g = []
        
        while(n_epochs > 0):
            data_not_exhausted = True
            while(data_not_exhausted):
                feed_dict = self.next_batch()
                for i in range(10):
                    loss_d.append(self.train_d(feed_dict))
                    print(i)
                for i in range(5):
                    loss_g.append(self.train_g(feed_dict))
                    print(i)
                n_epochs -= 1
                data_not_exhausted = self.pictures_done < 25
            print("Epochs_remaining :" + n_epochs)
        
        print(loss_d,loss_g)
            
               


    
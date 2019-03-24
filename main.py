# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:07:33 2019
@author: Parth
"""

import tensorflow as tf
import numpy as np
import os
from matplotlib.image import imread

tf.reset_default_graph()

#Global Variables
batch_size = 10
num_epochs = 1
lr = 0.0002
root = "F:\AI\Projects\CAN\smallimages"

def categ_init():
    temp = []
    for subdirs,dirs,files in os.walk(root):
        temp.append(subdirs)
    return temp[1:]



def categ_count_init(categs):
    temp = [0]
    for add in categs:
        for subdirs,dirs,files in os.walk(add):
            temp.append(len(files)+temp[-1])
    return(temp[1:])

categs = categ_init()    
categ_number_pictures = categ_count_init(categs)
len_dataset = categ_number_pictures[-1]



def next_batch_class_generator(batch_size,pictures_done):
    ind = 0
    classes = []
    
    for i in range(len(categ_number_pictures)):
        if pictures_done<=categ_number_pictures[i]:
           ind = i
           break

    if pictures_done>= len_dataset-batch_size:
        pictures_left = len_dataset-pictures_done
        for i in range(1,pictures_left):
            arr = [0]*27
            arr[ind] = 1
            classes.append(arr)
    else:
        for i in range(1,11):
            picture_number = pictures_done + i
            if categ_number_pictures[ind]>=picture_number:
                arr = [0]*27
                arr[ind] = 1
                classes.append(arr)
            else:
                arr = [0]*27
                arr[ind+1] = 1
                classes.append(arr)
    return classes

    
def next_batch_image_feed(batch_size,pictures_done):
    ind = 0
    images = []
    for i in range(len(categ_number_pictures)):
        if pictures_done<=categ_number_pictures[i]:
           ind = i
           break
    if pictures_done>= len_dataset-batch_size:
        pictures_left = len_dataset-pictures_done
        for i in range(1,pictures_left):
            picture_number = pictures_done + i
            for subdirs,dirs,files in os.walk(categs[ind]):
                add = subdirs + "/"
                images.append(imread(add+files[picture_number-categ_number_pictures[ind]]))
    else:
        for i in range(1,11):
            picture_number = pictures_done + i
            if categ_number_pictures[ind]>=picture_number:
                for subdirs,dirs,files in os.walk(categs[ind]):
                    add = subdirs + "/"
                    images.append(imread(add+files[picture_number-categ_number_pictures[ind]]))
            else:
                for subdirs,dirs,files in os.walk(categs[ind+1]):
                    add = subdirs + "/"
                    images.append(imread(add+files[picture_number-categ_number_pictures[ind+1]]))
    return images



def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        up = tf.keras.layers.UpSampling2D(size = (4,4))(x)
        
        l1 = tf.layers.conv2d_transpose(up, filters = 1024, kernel_size = (4,4), strides = (2,2), padding = "same")
        l1_rl = tf.nn.leaky_relu(l1,0.2) 
        
        l2 = tf.layers.conv2d_transpose(l1_rl, filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same")
        l2_rl = tf.nn.leaky_relu(l2,0.2) 
        
        l3 = tf.layers.conv2d_transpose(l2_rl, filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same")
        l3_rl = tf.nn.leaky_relu(l3,0.2)
        
        l4 = tf.layers.conv2d_transpose(l3_rl, filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same")
        l4_rl = tf.nn.leaky_relu(l4,0.2) 
        
        l5 = tf.layers.conv2d_transpose(l4_rl, filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same")
        l5_rl = tf.nn.leaky_relu(l5,0.2) 
        
        l6 = tf.layers.conv2d_transpose(l5_rl, filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same")
        l6_rl = tf.nn.leaky_relu(l6,0.2) 
    
        return l6_rl
    
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        conv1 = tf.layers.conv2d(x, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv1_lr = tf.nn.leaky_relu(conv1,0.2)
        
        conv2 = tf.layers.conv2d(conv1_lr, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv2_lr = tf.nn.leaky_relu(conv2,0.2)
        
        conv3 = tf.layers.conv2d(conv2_lr, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv3_lr = tf.nn.leaky_relu(conv3,0.2)
        
        conv4 = tf.layers.conv2d(conv3_lr, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv4_lr = tf.nn.leaky_relu(conv4,0.2)
        
        conv5 = tf.layers.conv2d(conv4_lr, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv5_lr = tf.nn.leaky_relu(conv5,0.2)
        
        conv6 = tf.layers.conv2d(conv5_lr, filters = 32, kernel_size = (4,4), strides = (2,2), padding = 'same')
        conv6_lr = tf.nn.leaky_relu(conv6,0.2)
        
        flat = tf.layers.flatten(conv6_lr)
        
        d_real_dense =  tf.layers.dense(flat,1024)
        d_real_dense_lr = tf.nn.leaky_relu(d_real_dense,0.2)
        d_real_prob = tf.layers.dense(d_real_dense_lr,1)
        
        d_class_dense_1 = tf.layers.dense(flat,1024)
        d_class_dense_lr = tf.nn.leaky_relu(d_class_dense_1,0.2)
        d_class_dense_2 = tf.layers.dense(d_class_dense_lr,1024)
        d_class_dense_2_lr = tf.nn.leaky_relu(d_class_dense_2,0.2)
        d_class_dense_3 = tf.layers.dense(d_class_dense_2_lr,512)
        d_class_dense_3_lr = tf.nn.leaky_relu(d_class_dense_3,0.2)
        d_class_prob = tf.layers.dense(d_class_dense_3_lr,27)
        
        return d_real_prob,d_class_prob


z = tf.placeholder(tf.float32, shape=(None, 1, 1, 2048))
x = tf.placeholder(tf.float32, shape = (None, 256,256,3))
tcv = tf.placeholder(tf.float32, shape = (None,27))
rcv = tf.placeholder(tf.float32, shape = (None,27))
isTrain = tf.placeholder(dtype=tf.bool)

G_z = generator(z, isTrain)
D_real_prob, D_real_class = discriminator(x, isTrain)
D_fake_prob, D_fake_class = discriminator(G_z, isTrain, reuse=True)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_prob, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_prob, labels=tf.zeros([batch_size, 1])))
D_real_class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_class, labels = rcv))
D_loss = D_loss_real + D_loss_fake + D_real_class_loss

class_cost_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_class, labels = tcv))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_prob, labels=tf.ones([batch_size, 1]))) + class_cost_generator

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)



def fit(n_epochs = 1):
    
    sess = tf.Session()
    saver = tf.train.Saver()
    if(1):
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        saver.restore(sess, "/tmp/model.ckpt")
               
    tot_epochs = n_epochs    
    loss_d = []
    loss_g = []

    while(n_epochs > 0):
        data_not_exhausted = True
        pictures_done = 0
        
        while(data_not_exhausted):
            
            imgs_nxt = next_batch_image_feed(batch_size,pictures_done)
            categs_nxt = next_batch_class_generator(batch_size,pictures_done)
            categs_nxt_fake = np.asarray(categs_nxt)*0.7
            
            Z= np.random.normal(size= [batch_size,1,1,2048])
            for i in range(len(categs_nxt)):
                Z[i,0,0,1024:] = np.nonzero(categs_nxt[i])[0]
            
            feed_dict = {z: Z, x: imgs_nxt, rcv: categs_nxt, tcv: categs_nxt_fake, isTrain: True} 
             
            loss_d_, _ = sess.run([D_loss, D_optim], feed_dict = feed_dict)
            loss_d.append(loss_d_)
            loss_g_, _ = sess.run([G_loss, G_optim], feed_dict = feed_dict)
            loss_g.append(loss_g_)
            
            pictures_done = pictures_done + batch_size
            data_not_exhausted = pictures_done < len_dataset
            
            if pictures_done%100 == 0:
                print("Pictures Done: " + str(pictures_done) + "/" + str(len_dataset) + " for epoch: " +str(6-n_epochs) +"/" +str(tot_epochs))
        n_epochs -= 1
        print("Epochs_done :" + str(n_epochs))
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
    print(len(loss_d),len(loss_g))
    

fit(5)


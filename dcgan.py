import os,path,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2


class BatchGenerator:
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.image = mnist.train.images
        self.label = mnist.train.labels

        self.image = np.reshape(self.image, [len(self.image), 28, 28])

    def getOne(self):
        idx = np.random.randint(0,len(self.image)-1)
        x,t = self.image[idx],self.label[idx]
        if color:
            x = np.expand_dims(x,axis=2)
            x = np.tile(x,(1,3))
        return x,t

    def getBatch(self,nBatch,color=True):
        idx = np.random.randint(0,len(self.image)-1,nBatch)
        x,t = self.image[idx],self.label[idx]
        if color:
            x = np.expand_dims(x,axis=3)
            x = np.tile(x,(1,1,3))
        return x,t

class DCGAN:
    def __init__(self,isTraining,imageSize,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 


    def buildGenerator(self,z,reuse=False,isTraining=True):
        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()
            h = z

            # fc1
            assert self.zdim == 7*7
            self.g_fc1_w, self.g_fc1_b = self._fc_variable([self.zdim,7*7*64],name="fc1")
            h = tf.matmul(h, self.g_fc1_w) + self.g_fc1_b
            h = tf.nn.relu(h)

            #
            h = tf.reshape(h,(self.nBatch,7,7,64))

            # deconv1
            self.g_deconv1_w, self.g_deconv1_b = self._deconv_variable([4,4,64,32],name="deconv1")
            h = self._deconv2d(h,self.g_deconv1_w, output_shape=[self.nBatch,14,14,32], stride=2) + self.g_deconv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm1")
            h = tf.nn.relu(h)

            # deconv2
            self.g_deconv2_w, self.g_deconv2_b = self._deconv_variable([4,4,32,3],name="deconv2")
            h = self._deconv2d(h,self.g_deconv2_w, output_shape=[self.nBatch,28,28,3], stride=2) + self.g_deconv2_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="gNorm2")
            #h = tf.nn.relu(h)

            # sigmoid
            #y = tf.nn.sigmoid(h)
            y = h

        return y

    def buildDiscriminator(self,y,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()

            h = y
            # conv1
            self.d_conv1_w, self.d_conv1_b = self._conv_variable([4,4,3,32],name="conv1")
            h = self._conv2d(h,self.d_conv1_w, stride=2) + self.d_conv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.d_conv2_w, self.d_conv2_b = self._conv_variable([4,4,32,64],name="conv2")
            h = self._conv2d(h,self.d_conv2_w, stride=2) + self.d_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
            h = tf.reshape(h,[self.nBatch,n_h*n_w*n_f])
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([n_h*n_w*n_f,1],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b
            
            d_logit = h
            d       = tf.nn.sigmoid(d_logit)

        return d,d_logit

    def buildModel(self):
        # define variables
        self.z      = tf.placeholder(tf.float32, [self.nBatch, self.zdim],name="z")

        self.y_real = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 3],name="image")
        self.y_fake = self.buildGenerator(self.z)
        self.y_sample = self.buildGenerator(self.z,reuse=True,isTraining=False)

        self.d_real, self.d_real_logit = self.buildDiscriminator(self.y_real)
        self.d_fake, self.d_fake_logit = self.buildDiscriminator(self.y_fake,reuse=True)

        # define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit,labels=tf.ones_like (self.d_real_logit)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit,labels=tf.zeros_like(self.d_fake_logit)))
        self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit,labels=tf.ones_like (self.d_fake_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_accuracy  = 0.5*(tf.reduce_mean(tf.cast(tf.greater(self.d_real,0.5),tf.float32))+tf.reduce_mean(tf.cast(tf.less(self.d_fake,0.5),tf.float32)))
        self.g_accuracy  = tf.reduce_mean(tf.cast(tf.greater(self.d_fake,0.5),tf.float32))

        # define optimizer
        self.g_optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        return

    def train(self,f_batch):
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15))
        self.sess = tf.Session(config=config)

        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        epoch = -1
        d_loss, g_loss = 1., 1.
        d_accuracy, g_accuracy = -1., -1.
        while True:
            epoch += 1

            batch_images,_ = f_batch(self.nBatch)
            batch_z        = np.random.uniform(-1,+1,[self.nBatch,self.zdim]).astype(np.float32)

            # update generator
            if d_loss > g_loss:
                _,d_loss,d_accuracy = self.sess.run([self.d_optimizer,self.d_loss,self.d_accuracy],feed_dict={self.z:batch_z, self.y_real:batch_images})
            else:
                _,g_loss,g_accuracy = self.sess.run([self.g_optimizer,self.g_loss,self.g_accuracy],feed_dict={self.z:batch_z, self.y_real:batch_images})
            if epoch%1000==0:
                print "%4d: loss(discri)=%.2e, loss(gener)=%.2e, accuracy(discri)=%.1f%%, accuracy(gener)=%.1f%%"%(epoch,d_loss,g_loss,d_accuracy*100., g_accuracy*100.)
                g_image = self.sess.run(self.y_sample,feed_dict={self.z:np.random.uniform(-1,+1,[self.nBatch,self.zdim]).astype(np.float32)})
                cv2.imwrite("log/img_%d.png"%epoch,g_image[0]*255.)
                #cv2.imwrite("log/img_%d.png"%epoch,batch_images[0]*255.)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:03:22 2020

@author: tmu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Add, Layer, Concatenate, Input, Dense, Flatten, Conv2D, AveragePooling2D, LeakyReLU, Reshape, UpSampling2D
from keras.initializers import RandomNormal
from keras.constraints import max_norm
from keras import backend as K
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MinibatchStdev(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)

	# perform the operation
	def call(self, inputs):
		# calculate the mean value for each pixel across channels
		mean = K.mean(inputs, axis=0, keepdims=True)
		# calculate the squared differences between pixel values and mean
		squ_diffs = K.square(inputs - mean)
		# calculate the average of the squared differences (variance)
		mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
		# add a small value to avoid a blow-up when we calculate stdev
		mean_sq_diff += 1e-8
		# square root of the variance (stdev)
		stdev = K.sqrt(mean_sq_diff)
		# calculate the mean standard deviation across each pixel coord
		mean_pix = K.mean(stdev, keepdims=True)
		# scale this up to be the size of one input feature map for each sample
		shape = K.shape(inputs)
		output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = K.concatenate([inputs, output], axis=-1)
		return combined

	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		# create a copy of the input shape as a list
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		# convert list to a tuple
		return tuple(input_shape)

class WeightedSum(Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = backend.variable(alpha, name='ws_alpha')

	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2)
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

def gentrue(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

def genfake(gmodel, latent_dim, n_samples):
    x_input = genlatent(latent_dim, n_samples)
    X = gmodel.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def genlatent(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def downsampling(model, filters=[16,32]):
    model = Conv2D(filters[0], (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(filters[1], (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = AveragePooling2D((2,2))(model)
    return model

def upsampling(model, filters=1024, upsam=True):
    model = Conv2D(filters, (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(filters, (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    if upsam == True:
        model = UpSampling2D((2,2))(model)
    return model

def decriminator(inputs, layers=6, keep_tensor=True):
    if keep_tensor == True:
        x = inputs
    else:
        x = Input(shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    layer = [[16,32], [32,64], [64,128], [128,256], [256,512], [512,512]]
    model = Conv2D(16, (1,1), strides=(1,1), padding='same', input_shape=(x.shape[1], x.shape[2], x.shape[3]))(x)
    model = LeakyReLU(alpha=0.2)(model)
    # Downsampling
    for i in range(layers):
        model = downsampling(model, layer[i])
    # Minibatch Stddev
    model = MinibatchStdev()(model)
    # Downsampling
    model = Conv2D(model.shape[-1]-1, (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(model.shape[-1], (4,4), strides=(4,4), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Flatten()(model)
    model = Dense(1)(model)
    # output = Model(x, model)
    # output.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return x, model

def decri_block(gmodel_list, n_blocks=7, keep_tensor=True):
    model_list = list()
    for i in range(n_blocks):
        de_input, de_output = decriminator(gmodel_list[i][1], layers=i+1, keep_tensor=keep_tensor)
        model_list.append([de_input, de_output])
    return model_list

def generator(flat_img, dim=4, layers=6):
    layer = [512, 256, 128, 64, 32, 16]
    x = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    latent = Input(shape=(flat_img,))
    model = Dense(512 * dim * dim, kernel_initializer=x, kernel_constraint=const)(latent)
    model = Reshape((dim, dim, 512))(model)
    # Upsampling
    model = Conv2D(512, (4,4), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(512, (3,3), strides=(1,1), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = UpSampling2D((2,2))(model)
    # Upsampling
    for i in range(layers):
        if i == layers-1:
            model = upsampling(model, layer[i], upsam=False)
        else:
            model = upsampling(model, layer[i])
    # Linear
    model = Conv2D(1, (1,1), strides=(1,1), activation='linear', padding='same')(model)
    return latent, model

def gen_block(latent_dim, dim, n_blocks=6):
    model_list = list()
    for i in range(n_blocks):
        gen_input, gen_output = generator(latent_dim, dim, layers=i+1)
        model_list.append([gen_input, gen_output])
    return model_list

# scale images to perferred size
def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps-1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)

# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs=100, n_batch=128, fadein=False):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    for i in range(n_steps):
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        X_real, y_real = gentrue(dataset, half_batch)
        X_fake, y_fake = genfake(g_model, latent_dim, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        z_input = genlatent(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        print('%d/%d, dloss1 = %.3f, dloss2 = %.3f, loss = %.3f' % (i+1, n_steps, d_loss1, d_loss2, g_loss))

# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    X, _ = genfake(g_model, latent_dim, n_samples)
    X = (X-X.min())/(X.max()-X.min())
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(square, square, i+1)
        plt.axis('off')
        plt.imshow(X[i,:,:,0])
    filename1 = '/home/kurtlin2012/PGGAN/plot/plot_%s.png' % (name)
    plt.savefig(filename1, bbox_inches='tight', dpi=300)
    plt.close()
    filename2 = '/home/kurtlin2012/PGGAN/model/model_%s.h5' % (name)
    g_model.save(filename2)
    print('Plot and model saved: %s and %s' % (filename1, filename2))

def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    g_normal, d_normal, gan_normal = g_models[0], d_models[0], gan_models[0]
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal, latent_dim)
    for i in range(len(g_models)):
        g_normal, g_fadein = g_models[i], g_models[i]
        d_normal, d_fadein = d_models[i], d_models[i]
        gan_normal, gan_fadein = gan_models[i], gan_models[i]
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein, latent_dim)
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal, latent_dim)
      
# Main
gmodels = gen_block(100, 4, 6)
dmodels = decri_block(gmodels, 6, keep_tensor=False)
ganmodels = decri_block(gmodels, 6, keep_tensor=True)
g_models = list()
d_models = list()
gan_models = list()
for i in range(6):
    gtemp = Model(gmodels[i][0], gmodels[i][1])
    gtemp.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, beta_1=0.5))
    g_models.append(gtemp)
    del gtemp
    dtemp = Model(dmodels[i][0], dmodels[i][1])
    dtemp.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, beta_1=0.5))
    d_models.append(dtemp)
    del dtemp
    gantemp = Model(gmodels[i][0], ganmodels[i][1])
    gantemp.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, beta_1=0.5))
    gan_models.append(gantemp)
    del gantemp
    
# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 6
# size of the latent space
# latent_dim = 100
# load image data
data_ori = np.load('/work/kurtlin2012/Data/CTAori_train1.npy')
data_ori[data_ori < 0] = 0
dataset = np.zeros([data_ori.shape[0] * 7, data_ori.shape[1], data_ori.shape[2], data_ori.shape[4]], dtype='float32')
for i in range(data_ori.shape[0]):
    for j in range(13, 20):
        dataset[i*7+(j-13),:,:,:] = data_ori[i,:,:,j,:]
    if (i+1) % 20 == 0:
        print('X Processing: ' + str(i+1) + '/200')
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 8, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
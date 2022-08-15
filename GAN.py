# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:12:57 2021

@author: Ching-Ting Lin
"""

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, BatchNormalization,
                                     UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

file_path = ''
out_path = ''

def dbconv(inputs, filters, ker_ini=None):
    model = Conv2D(filters=filters, kernel_size=[3,3], padding='same', kernel_initializer=ker_ini)(inputs)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(filters=filters, kernel_size=[3,3], padding='same', kernel_initializer=ker_ini)(model)
    model = BatchNormalization()(model)
    out = Activation('relu')(model)
    return out

# U-Net without skip connection
def generator(img, filters):
    input_shape = [img.shape[1], img.shape[2], img.shape[3]]
    inputs = Input(input_shape)
    model1 = dbconv(inputs, filters=filters, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    model2 = MaxPooling2D(pool_size=(2,2))(model1)
    model2 = dbconv(model2, filters=filters*2, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    model3 = MaxPooling2D(pool_size=(2,2))(model2)
    model3 = dbconv(model3, filters=filters*4, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    model4 = MaxPooling2D(pool_size=(2,2))(model3)
    model4 = dbconv(model4, filters=filters*8, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    modelb3 = Conv2DTranspose(filters=filters*4, kernel_size=[3,3], strides=[2,2], padding='same')(model4)
    modelb3 = dbconv(modelb3, filters=filters*4, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    modelb2 = Conv2DTranspose(filters=filters*2, kernel_size=[3,3], strides=[2,2], padding='same')(modelb3)
    modelb2 = dbconv(modelb2, filters=filters*2, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    modelb1 = Conv2DTranspose(filters=filters, kernel_size=[3,3], strides=[2,2], padding='same')(modelb2)
    modelb1 = dbconv(modelb1, filters=filters, ker_ini=RandomNormal(mean=1.0, stddev=0.5, seed=100))
    out = Conv2D(filters=1, kernel_size=[1,1], padding='same', activation='relu')(modelb1)
    
    model = Model(inputs, out)
    return model

# U-Net
def discriminator(img, filters):
    input_shape = [img.shape[1], img.shape[2], img.shape[3]]
    inputs = Input(input_shape)
    model1 = dbconv(inputs, filters=filters)
    model2 = MaxPooling2D(pool_size=(2,2))(model1)
    model2 = dbconv(model2, filters=filters*2)
    model3 = MaxPooling2D(pool_size=(2,2))(model2)
    model3 = dbconv(model3, filters=filters*4)
    model4 = MaxPooling2D(pool_size=(2,2))(model3)
    model4 = dbconv(model4, filters=filters*8)
    modelb3 = UpSampling2D(size=(2,2))(model4)
    modelb3 = Concatenate(axis=-1)([modelb3, model3])
    modelb3 = dbconv(modelb3, filters=filters*4)
    modelb2 = UpSampling2D(size=(2,2))(modelb3)
    modelb2 = Concatenate(axis=-1)([modelb2, model2])
    modelb2 = dbconv(modelb2, filters=filters*2)
    modelb1 = UpSampling2D(size=(2,2))(modelb2)
    modelb1 = Concatenate(axis=-1)([modelb1, model1])
    modelb1 = dbconv(modelb1, filters=filters)
    out = Conv2D(filters=1, kernel_size=[1,1], padding='same', activation='sigmoid')(modelb1)
    
    model = Model(inputs, out)
    return model


# Square Error
def generator_loss_fn(y_true, y_fake):
    return K.mean(K.square(y_true - y_fake), keepdims=False)


# Dice Loss (or Categorial Crossentropy)
def discriminator_loss_fn(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = 2 * (intersection + smooth) / (union + smooth)
    return (1 - dice)


# Main GAN network
class KGan(tf.keras.Model):
    def __init__(self, generator, discriminator_x, discriminator_y):
        super(KGan, self).__init__()
        self.gen = generator
        self.disc_x = discriminator_x
        self.disc_y = discriminator_y
    
    def compile(self, gen_optim, disc_x_optim, disc_y_optim, gen_loss, disc_loss):
        super(KGan, self).compile()
        self.gen_optimizer = gen_optim
        self.disc_x_optimizer = disc_x_optim
        self.disc_y_optimizer = disc_y_optim
        self.generator_loss_fn = gen_loss
        self.discriminator_loss_fn = disc_loss
        
    def train_step(self, batch_data):
        real_x, real_y, real_z = batch_data     # X for NCCT, Y for CTA, Z for Vessel
        
        with tf.GradientTape(persistent=True) as tape:
            # Generator
            fake_y = self.gen(real_x, training=True)
            
            # Discriminator
            disc_fake_y = self.disc_x(fake_y, training=True)
            disc_real_y = self.disc_y(real_y, training=True)
            
            # Loss function
            gen_loss = self.generator_loss_fn(real_y, fake_y)
            disc_loss_x = self.discriminator_loss_fn(disc_fake_y, real_z)
            disc_loss_y = self.discriminator_loss_fn(disc_real_y, real_z)
        
        gen_x_grads = tape.gradient(gen_loss, self.gen.trainable_variables)
        
        disc_x_grads = tape.gradient(disc_loss_x, self.disc_x.trainable_variables)
        disc_y_grads = tape.gradient(disc_loss_y, self.disc_y.trainable_variables)
        
        self.gen_optimizer.apply_gradients(
            zip(gen_x_grads, self.gen.trainable_variables))
        
        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_grads, self.disc_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.disc_y.trainable_variables))
        
        return {
            "Gen_loss": gen_loss,
            "Discx_loss": disc_loss_x,
            "Discy_loss": disc_loss_y}

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test.take(self.num_img)):
            prediction = self.model.gen(img)[0].numpy()
            img = (img[0]).numpy()
            img[img < 0] = 0
            img[img > 200] = 200
            prediction[prediction < 0] = 0
            prediction[prediction > 200] = 200
            # prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            # img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img[...,0], cmap='gray')
            ax[i, 1].imshow(prediction[...,0], cmap='gray')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        
        plt.show()
        plt.savefig(out_path + '/generated_{epoch:0>3d}.png'.format(epoch = epoch+1), dpi=350,
                    bbox_inches='tight')
        plt.close()
        

X = np.load(f'{file_path}/X.npy').astype('float32')   # NCCT images
print('Load X: ' + str(X.shape))
y = np.load(f'{file_path}/y.npy').astype('float32')   # CTA images
print('Load y: ' + str(y.shape))
z = np.load(f'{file_path}/z.npy')   # Vessel maps
z = np.expand_dims(z[...,0], axis=-1).astype('float32')
print('Load z: ' + str(z.shape))

val_num = int(X.shape[0] // 10)
X_train = X[:-val_num]
X_test = X[-val_num:]
del X

y_train = y[:-val_num]
y_test = y[-val_num:]
del y

z_train = z[:-val_num]
z_test = z[-val_num:]
del z

gen_X2Y = generator(X_train, 16)
disc_X2Z = discriminator(X_train, 16)
disc_Y2Z = discriminator(y_train, 16)

gan_model = KGan(gen_X2Y, disc_X2Z, disc_Y2Z)
gan_model.compile(gen_optim = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5), 
                  disc_x_optim = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5), 
                  disc_y_optim = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5), 
                  gen_loss = generator_loss_fn, 
                  disc_loss = discriminator_loss_fn)

plotter = GANMonitor()
checkpoint_filepath = out_path + "/kgan_{epoch:0>3d}_{Gen_loss:.4f}_{Discx_loss:.4f}_{Discy_loss:.4f}.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath)

train = tf.data.Dataset.from_tensor_slices((X_train, y_train, z_train)).batch(2)
test = tf.data.Dataset.from_tensor_slices((X_test)).shuffle(100).batch(2)

gan_model.fit(train, epochs=50, callbacks=[plotter, model_checkpoint])

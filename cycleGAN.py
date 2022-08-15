# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:09:49 2020

@author: Ching-Ting Kurt Lin
"""

import os 
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, 
                                     BatchNormalization, Concatenate, LeakyReLU, Conv2DTranspose)
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import pydicom
from skimage.transform import resize
import time
from IPython.display import clear_output
from sklearn.utils import shuffle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

out_path = ''
X = np.load('')
y = np.load('')

val_num = X.shape[0] // 10
X_train = X[:-val_num]
X_test = X[-val_num:]
y_train = y[:-val_num]
y_test = y[-val_num:]
test = tf.data.Dataset.from_tensor_slices((X_test)).batch(2)
del X, y


def dbconv(model, filters):
    for i in range(2):
        model = Conv2D(filters = filters, kernel_size = [3, 3], padding = 'same', kernel_initializer=RandomNormal)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
    return model

def gen_block(img, model = None):
    if model == None:
        inputs = Input(shape = [img.shape[1], img.shape[2], img.shape[3]])
    else:
        inputs = model
    enc1 = dbconv(inputs, 32)
    enc2 = MaxPooling2D(pool_size=(2, 2))(enc1)
    enc2 = dbconv(enc2, 64)
    enc3 = MaxPooling2D(pool_size=(2, 2))(enc2)
    enc3 = dbconv(enc3, 128)
    enc4 = MaxPooling2D(pool_size=(2, 2))(enc3)
    enc4 = dbconv(enc4, 256)
    
    dec1 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(enc4)
    # dec1 = Concatenate(axis=-1)([dec1, enc3])
    dec1 = dbconv(dec1, 128)
    dec2 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(dec1)
    # dec2 = UpSampling2D(size=(2, 2))(dec1)
    dec2 = dbconv(dec2, 64)
    dec3 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(dec2)
    # dec3 = UpSampling2D(size=(2, 2))(dec2)
    dec3 = dbconv(dec3, 32)
    
    out = Conv2D(filters = 1, kernel_size = [3, 3], padding = 'same', kernel_initializer=RandomNormal)(dec3)
    return inputs, out


def generator(img):
    x1in, x1out = gen_block(img, None)
    x2in, x2out = gen_block(img, x1out)
    model = Model(x1in, x2out)
    return model
    

def discriminator(img):
    inputs = Input(shape = [img.shape[1], img.shape[2], img.shape[3]])
    des1 = Conv2D(filters = 32, kernel_size = [3, 3], padding = 'same', kernel_initializer='glorot_normal')(inputs)
    des1 = LeakyReLU(alpha=0.3)(des1)
    des2 = Conv2D(filters = 64, kernel_size = [3, 3], padding = 'same', kernel_initializer='glorot_normal')(des1)
    des2 = BatchNormalization()(des2)
    des2 = LeakyReLU(alpha=0.3)(des2)
    des3 = Conv2D(filters = 128, kernel_size = [3, 3], padding = 'same', kernel_initializer='glorot_normal')(des2)
    des3 = BatchNormalization()(des3)
    des3 = LeakyReLU(alpha=0.3)(des3)
    des4 = Conv2D(filters = 256, kernel_size = [3, 3], padding = 'same', kernel_initializer='glorot_normal')(des3)
    des4 = BatchNormalization()(des4)
    des4 = LeakyReLU(alpha=0.3)(des4)
    
    out = Conv2D(filters = 1, kernel_size = [3, 3], padding = 'same', kernel_initializer='glorot_normal')(des4)
    
    model = Model(inputs, out)
    return model

adv_loss_fn = tf.keras.losses.MeanSquaredError()
cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
identity_loss_fn = tf.keras.losses.MeanAbsoluteError()
lambda_cycle = 10.0
lambda_identity = 0.5

def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

class CycleGan(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    def train_step(self, batch_data):
        real_x, real_y = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)

            cycled_x = self.gen_F(fake_y, training=True)
            cycled_y = self.gen_G(fake_x, training=True)

            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_lo": total_loss_G,
            "F_lo": total_loss_F,
            "DX_lo": disc_X_loss,
            "DY_lo": disc_Y_loss,
        }

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test.take(self.num_img)):
            
            prediction = self.model.gen_G(img)[0].numpy()
            #prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            #img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            img = (img[0]).numpy()
            
            ax[i, 0].imshow(img, cmap='gray')
            ax[i, 1].imshow(prediction, cmap='gray')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            #prediction = keras.preprocessing.image.array_to_img(prediction)
            #prediction.save(
            #    "/work/kurtlin2012/Cyclegan/Predict/generated_img_t_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            #)
        
        plt.show()
        plt.savefig(out_path + "/generated_img_{epoch}.png".format(epoch=epoch + 1), dpi=350,
                    bbox_inches='tight')
        plt.close()
            

gen_A2B = generator(X_train)
gen_B2A = generator(y_train)
disc_A2B = discriminator(X_train)
disc_B2A = discriminator(y_train)

cycle_gan_model = CycleGan(generator_G=gen_A2B, generator_F=gen_B2A, discriminator_X=disc_A2B, discriminator_Y=disc_B2A)
cycle_gan_model.compile(
    gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn)

plotter = GANMonitor()
checkpoint_filepath = out_path + "/cyclegan_{epoch:03d}_{G_lo:.5f}_{F_lo:.5f}.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(2)

cycle_gan_model.fit(train, epochs=50, callbacks=[plotter, model_checkpoint_callback])

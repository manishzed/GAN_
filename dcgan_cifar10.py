# -*- coding: utf-8 -*-
"""dcgan_CIFAR10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OJxt75KHiG_ca1iZKMioLwiEQFrJ1pb5
"""



import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from math import ceil
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, ReLU, LeakyReLU, Dense
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization   
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import initializers

def build_cifar10_discriminator(ndf=64, image_shape=(32, 32, 3)):
    """ Builds CIFAR10 DCGAN Discriminator Model
    PARAMS
    ------
    ndf: number of discriminator filters
    image_shape: 32x32x3
    RETURN
    ------
    D: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.02)

    D = Sequential()

    # Conv 1: 16x16x64
    D.add(Conv2D(ndf, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init,
                 input_shape=image_shape))
    D.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    D.add(Conv2D(ndf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 3: 4x4x256
    D.add(Conv2D(ndf*4, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 4:  2x2x512
    D.add(Conv2D(ndf*8, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Flatten: 2x2x512 -> (2048)
    D.add(Flatten())

    # Dense Layer
    D.add(Dense(1, kernel_initializer=init))
    D.add(Activation('sigmoid'))

    print("\nDiscriminator")
    D.summary()

    return D


def build_cifar10_generator(ngf=64, z_dim=128):
    """ Builds CIFAR10 DCGAN Generator Model
    PARAMS
    ------
    ngf: number of generator filters
    z_dim: number of dimensions in latent vector
    RETURN
    ------
    G: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.02)

    G = Sequential()

    # Dense 1: 2x2x512
    G.add(Dense(2*2*ngf*8, input_shape=(z_dim, ),
        use_bias=True, kernel_initializer=init))
    G.add(Reshape((2, 2, ngf*8)))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 1: 4x4x256
    G.add(Conv2DTranspose(ngf*4, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    G.add(Conv2DTranspose(ngf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 3: 16x16x64
    G.add(Conv2DTranspose(ngf, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 4: 32x32x3
    G.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(Activation('tanh'))

    print("\nGenerator")
    G.summary()

    return G


def get_data():
    # load cifar10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert train and test data to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # scale train and test data to [-1, 1]
    X_train = (X_train / 255) * 2 - 1
    X_test = (X_train / 255) * 2 - 1

    return X_train, X_test


def plot_images(images, filename):
    h, w, c = images.shape[1:]
    grid_size = ceil(np.sqrt(images.shape[0]))
    images = (images.reshape(grid_size, grid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(grid_size*h, grid_size*w, c))
    plt.figure(figsize=(16, 16))
    plt.imsave(filename, images)


def plot_losses(losses_d, losses_g, filename):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    axes[0].plot(losses_d)
    axes[1].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train(ndf=64, ngf=64, z_dim=100, lr_d=2e-4, lr_g=2e-4, epochs=100,
          batch_size=128, epoch_per_checkpoint=1, n_checkpoint_images=36):

    X_train, _ = get_data()
    image_shape = X_train[0].shape
    print("image shape {}, min val {}, max val {}".format(
        image_shape, X_train[0].min(), X_train[0].max()))

    # plot real images for reference
    #plot_images(X_train[:n_checkpoint_images], "real_images.png")

    # build models
    D = build_cifar10_discriminator(ndf, image_shape)
    G = build_cifar10_generator(ngf, z_dim)

    # define Discriminator's optimizer
    D.compile(Adam(lr=lr_d, beta_1=0.5), loss='binary_crossentropy',
              metrics=['binary_accuracy'])

    # define D(G(z)) graph for training the Generator
    D.trainable = False
    z = Input(shape=(z_dim, ))
    D_of_G = Model(inputs=z, outputs=D(G(z)))

    # define Generator's Optimizer
    D_of_G.compile(Adam(lr=lr_g, beta_1=0.5), loss='binary_crossentropy',
                   metrics=['binary_accuracy'])

    # get labels for computing the losses
    labels_real = np.ones(shape=(batch_size, 1))
    labels_fake = np.zeros(shape=(batch_size, 1))

    losses_d, losses_g = [], []

    # fix a z vector for training evaluation
    z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_images, z_dim))

    # training loop
    for e in range(epochs):
        print("Epoch {}".format(e))
        for i in range(len(X_train) // batch_size):
            #saving weights
            print("epochs:{}, i:{}".format(e, i))
            #D_of_G.save_weights('./model/gan_.h5')
            #G.save_weights('./model/generator_.h5')
            #D.save_weights('./model/discriminator_.h5')

            # update Discriminator weights
            D.trainable = True

            # Get real samples
            real_images = X_train[i*batch_size:(i+1)*batch_size]
            loss_d_real = D.train_on_batch(x=real_images, y=labels_real)[0]

            # Fake Samples
            z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            fake_images = G.predict_on_batch(z)
            loss_d_fake = D.train_on_batch(x=fake_images, y=labels_fake)[0]

            # Compute Discriminator's loss
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            # update Generator weights, do not update Discriminator weights
            D.trainable = False
            loss_g = D_of_G.train_on_batch(x=z, y=labels_real)[0]

        losses_d.append(loss_d)
        losses_g.append(loss_g)
        """
        if (e % epoch_per_checkpoint) == 0:
            print("loss_d={:.5f}, loss_g={:.5f}".format(loss_d, loss_g))
            fake_images = G.predict(z_fixed)
            print("\tPlotting images and losses")
            plot_images(fake_images, "fake_images_e{}.png".format(e))
            plot_losses(losses_d, losses_g, "losses.png")
        """

train()

!rm -rf model res_v1 res_v2

!mkdir model res_v1 res_v2

#testing and loading weights------------------------------------------------------

def build_cifar10_discriminator(ndf=64, image_shape=(32, 32, 3)):
    """ Builds CIFAR10 DCGAN Discriminator Model
    PARAMS
    ------
    ndf: number of discriminator filters
    image_shape: 32x32x3
    RETURN
    ------
    D: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.02)

    D = Sequential()

    # Conv 1: 16x16x64
    D.add(Conv2D(ndf, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init,
                 input_shape=image_shape))
    D.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    D.add(Conv2D(ndf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 3: 4x4x256
    D.add(Conv2D(ndf*4, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 4:  2x2x512
    D.add(Conv2D(ndf*8, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Flatten: 2x2x512 -> (2048)
    D.add(Flatten())

    # Dense Layer
    D.add(Dense(1, kernel_initializer=init))
    D.add(Activation('sigmoid'))

    print("\nDiscriminator")
    D.summary()

    return D


def build_cifar10_generator(ngf=64, z_dim=128):
    """ Builds CIFAR10 DCGAN Generator Model
    PARAMS
    ------
    ngf: number of generator filters
    z_dim: number of dimensions in latent vector
    RETURN
    ------
    G: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.02)

    G = Sequential()

    # Dense 1: 2x2x512
    G.add(Dense(2*2*ngf*8, input_shape=(z_dim, ),
        use_bias=True, kernel_initializer=init))
    G.add(Reshape((2, 2, ngf*8)))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 1: 4x4x256
    G.add(Conv2DTranspose(ngf*4, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    G.add(Conv2DTranspose(ngf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 3: 16x16x64
    G.add(Conv2DTranspose(ngf, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 4: 32x32x3
    G.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(Activation('tanh'))

    print("\nGenerator")
    G.summary()

    return G

#load model weights
import os

images_saved = 0
LATENT_DIM = 32
CHANNELS = 3
CONTROL_SIZE_SQRT = 6
WIDTH = 32
HEIGHT = 32

ndf=64
ngf=64
z_dim=100
lr_d=2e-4
lr_g=2e-4
epochs=3
batch_size=128
epoch_per_checkpoint=1
n_checkpoint_images=36
g_file_name = '/content/model/generator_.h5'
d_file_name = '/content/model/discriminator_.h5'

G_ = build_cifar10_generator(ngf=64, z_dim=100)
D_ = build_cifar10_discriminator(ndf=64, image_shape=(32, 32, 3))



G_.load_weights(g_file_name)

D_.load_weights(d_file_name)

D_.summary()
G_.summary()

#inference
import cv2
from PIL import Image
images_saved = 0
LATENT_DIM = 100
CHANNELS = 3
#number of image to generate
CONTROL_SIZE_SQRT = 5
WIDTH =32
HEIGHT=32

RES_DIR_v1= 'res_v1'
RES_DIR_v2= 'res_v2'

FILE_PATH = '%s/generated_%d.png'
control_vectors_ = np.random.normal(size=(CONTROL_SIZE_SQRT, LATENT_DIM)) / 2
control_generated_ = G_.predict(control_vectors_)
step=11098765
fig = plt.figure(figsize=(4, 4))
control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
print("11111", control_generated_.shape[0])
print("showing colored image")
#save colored image individually
for i in range(CONTROL_SIZE_SQRT):
      im = Image.fromarray(np.uint8(control_generated_[i, :, :, :] * 255))
      im.save(FILE_PATH % (RES_DIR_v1, images_saved))
      images_saved += 1
      
print("showing gray image")
#save single gray image at each 50 epochs 
for i in range(control_generated_.shape[0]):
    #plt.subplot(4, 4, i+1)
    #plt.imshow(control_generated_[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.imsave('res_v2/image_at_epoch_{:04d}-{}.png'.format(step, i), control_generated_[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    #plt.axis('off')


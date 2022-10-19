from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X
 
# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load and prepare training images
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = numpy.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X


from numpy import load
# load dataset
A_data, B_data = load_real_samples('./horse2zebra_256.npz')
#print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)

model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

# plot A->B->A
A_real = select_sample(A_data, 1)
#A_real =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_20.jpg"
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
#B_real =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_80.jpg"
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)





#for custom image manual------------------------------------------------------------yyyyyyyyyyyyyy-----
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)

#g_model_AtoB_018992 and g_model_BtoA_018992

model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_updated_v1/g_model_AtoB_024717.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_updated_v1/g_model_BtoA_024717.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)





#for custom image manual------------------------------------------------------------xxxxxxxxxxxxxxxxxxx-----
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)

#g_model_AtoB_018992 and g_model_BtoA_018992

model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_100e/g_model_AtoB_009496.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_100e/g_model_BtoA_009496.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)








#on after 83000 + iterations 
#for custom image manual------------------------------------------------------------00000000000000000000000000000000000000000
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy

# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)

#g_model_AtoB_018992 and g_model_BtoA_018992

model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_83000/g_model_AtoB_024927.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_83000/g_model_BtoA_024927.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)




#on after 47000 + iterations 
#for custom image manual-------------------------------------------------------------111111111111111111
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy

# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)

#g_model_AtoB_018992 and g_model_BtoA_018992

model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_47000/model/g_model_AtoB_028488.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_trained_after_47000/model/g_model_BtoA_028488.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)



#on 47480 iterations again

#for custom image manual-------------------------------------------------------------222222222222
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy

# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()

# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)
             
model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_v5/model_47000/model_47000/g_model_AtoB_047480.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_v5/model_47000/model_47000/g_model_BtoA_047480.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)




#on 47480 iterations

#for custom image manual-------------------------------------------------------------33333333333333333
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy

# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()

# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)
             
model_AtoB = tf.keras.models.load_model(r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/g_model_AtoB_047480.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/g_model_BtoA_047480.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)















#on 5935 iterationss

#for custom image manual-------------------------------------------------------------444444444444444444444444444
# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
import numpy

# function to load one image for translation
def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	# scale pixel
	pixels = (pixels - 127.5) / 127.5
	return pixels

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}

#model_AtoB = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./model_v5/model_47000/model_47000/g_model_BtoA_047480.h5', cust)

#model_AtoB = tf.keras.models.load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = tf.keras.models.load_model('./g_model_BtoA_047480.h5', cust)
             
model_AtoB = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_v4/g_model_AtoB_005935.h5", cust)                         
model_BtoA = tf.keras.models.load_model(r"D:/softweb/cycleGAN_horse2zebra/model_v4/g_model_BtoA_005935.h5", cust)
# plot A->B->A
#A_real = select_sample(A_data, 1)
A_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
A_real_ =load_image(A_real_)

B_generated  = model_AtoB.predict(A_real_)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real_, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 1)
B_real_ =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
B_real_ =load_image(B_real_)


A_generated  = model_BtoA.predict(B_real_)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real_, A_generated, B_reconstructed)









from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
 
# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X
 
# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()


# load dataset
A_data, B_data = load_real_samples('horse2zebra_256.npz')
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
#model_AtoB = load_model('./g_model_AtoB_047480.h5', cust)
#model_BtoA = load_model('./g_model_BtoA_047480.h5', cust)

model_AtoB = load_model(r"D:\softweb\cycleGAN_horse2zebra\model_v5\model_47000\model_47000/g_model_AtoB_047480.h5", cust)
model_BtoA = load_model(r"D:\softweb\cycleGAN_horse2zebra\model_v5\model_47000\model_47000/g_model_BtoA_047480.h5", cust)


# plot A->B->A
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)









# example of using saved cyclegan models for image translation
from numpy import load
from numpy import expand_dims
from tensorflow.keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot
 
# load an image to the preferred size
def load_image(filename, size=(256,256)):
	# load and resize the image
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# transform in a sample
	pixels = expand_dims(pixels, 0)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	return pixels
 
# load the image
img_A =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testA/n02381460_120.jpg"
image_src_A = load_image(img_A)
# load the model
cust = {'InstanceNormalization': InstanceNormalization}
#model_AtoB = load_model('g_model_AtoB_047480.h5', cust)
model_AtoB = load_model(r"D:\softweb\cycleGAN_horse2zebra\model_v5\model_47000\model_47000/g_model_AtoB_047480.h5", cust)
# translate image
image_tar_A = model_AtoB.predict(image_src_A)
# scale from [-1,1] to [0,1]
image_tar_A = (image_tar_A + 1) / 2.0
# plot the translated image
pyplot.imshow(image_tar_A[0])
pyplot.show()

img_B =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/testB/n02391049_4890.jpg"
image_src_B= load_image(img_B)
cust = {'InstanceNormalization': InstanceNormalization}
#model_BtoA = load_model('g_model_BtoA_047480.h5', cust)
model_BtoA = load_model(r"D:\softweb\cycleGAN_horse2zebra\model_v5\model_47000\model_47000/g_model_BtoA_047480.h5", cust)

image_tar_B = model_BtoA.predict(image_src_B)

# scale from [-1,1] to [0,1]
image_tar_B = (image_tar_B + 1) / 2.0
# plot the translated image
pyplot.imshow(image_tar_B[0])
pyplot.show()









#testing pretrained kaggle model-----------------------------------------------

from IPython.display import clear_output as clear
!pip install tensorflow-addons
clear()

# Common
import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from random import random

# Data
import tensorflow.image as tfi
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Model Layers
from keras.layers import ReLU
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import ZeroPadding2D
from keras.layers import Conv2DTranspose
##from tensorflow_addons.layers import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# Model Functions
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.initializers import RandomNormal

# Optimizers
from tensorflow.keras.optimizers import Adam

# Loss
from keras.losses import BinaryCrossentropy

# Model Viz
from tensorflow.keras.utils import plot_model
def ResidualBlock(filters, layer, index):
#     init = RandomNormal(stddev=0.02)
    
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, name="Block_{}_Conv1".format(index))(layer)
    x = InstanceNormalization(axis=-1, name="Block_{}_Normalization1".format(index))(x)
    x = ReLU(name="Block_{}_ReLU".format(index))(x)
    
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, name="Block_{}_Conv2".format(index))(x)
    x = InstanceNormalization(axis=-1, name="Block_{}_Normalization2".format(index))(x)
    
    x = concatenate([x, layer], name="Block_{}_Merge".format(index))
    
    return x

def downsample(filters, layer, size=3, strides=2, activation=None, index=None, norm=True):
    x = Conv2D(filters, kernel_size=size, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False, name="Encoder_{}_Conv".format(index))(layer)
    if norm:
        x = InstanceNormalization(axis=-1, name="Encoder_{}_Normalization".format(index))(x)
    if activation is not None:
        x = Activation(activation, name="Encoder_{}_Activation".format(index))(x)
    else:
        x = LeakyReLU( name="Encoder_{}_LeakyReLU".format(index))(x)
    return x

def upsample(filters, layer, size=3, strides=2, index=None):
    x = Conv2DTranspose(filters, kernel_size=size, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False, name="Decoder_{}_ConvT".format(index))(layer)
    x = InstanceNormalization(axis=-1, name="Decoder_{}_Normalization".format(index))(x)
    x = ReLU( name="Encoder_{}_ReLU".format(index))(x)
    return x


def Generator(n_resnet=9, name="Generator"):
    
    inp_image = Input(shape=(SIZE, SIZE, 3), name="InputImage")         # 256 x 256 x3
    
    x = downsample(64, inp_image, size=7, strides=1, index=1)           # 256 x 256 x 64
    x = downsample(128, x, index=2)                                     # 128 x 128 x 128
    x = downsample(256, x, index=3)                                     # 64 x 64 x 256
    
    for i in range(n_resnet):
        x = ResidualBlock(256, x, index=i+4)                             # (64 x 64 x 256) x n_resnet
    
    x = upsample(128, x, index=13)                                       # 128 x 128 x 128
    x = upsample(64, x, index=14)                                        # 256 x 256 x 64
    x = downsample(3, x, size=7, strides=1, activation='tanh', index=15) # 256 x 256 x 3
    
    model = Model(
        inputs=inp_image,
        outputs=x,
        name=name
    )
    return model
def Discriminator(name='Discriminator'):
    init = RandomNormal(stddev=0.02)
    src_img = Input(shape=(SIZE, SIZE, 3), name="InputImage")     # 256 x 256 x 3
    x = downsample(64, src_img, size=4, strides=2, index=1, norm=False) # 128 x 128 x 64
    x = downsample(128, x, size=4, strides=2, index=2)            # 64 x 64 x 128
    x = downsample(256, x, size=4, strides=2, index=3)            # 32 x 32 x 256
    x = downsample(512, x, size=4, strides=2, index=4)            # 16 x 16 x 512
    x = downsample(512, x, size=4, strides=2, index=5)            # 8 x 8 x 512  # we can try out a different architecture with zero padding layer.
    patch_out = Conv2D(1, kernel_size=4, padding='same', kernel_initializer=init, use_bias=False)(x) # 8 x 8 x 1
    
    model = Model(
        inputs=src_img,
        outputs=patch_out,
        name=name
    )
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        loss_weights=[0.5]
    )
    return model
# let's create our generators.
g_AB = Generator(name="GeneratorAB")
g_BA = Generator(name="GeneratorBA")

root_horse_path =r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/trainA"
root_zebra_path = r"C:/Users/manish.kumar/Desktop/GAN/cycleGAN_horse2zebra/horse2zebra/horse2zebra/trainB"
horse_paths = sorted(glob(root_horse_path + '/*.jpg'))[:1000]
zebra_paths = sorted(glob(root_zebra_path + '/*.jpg'))[:1000]
#In total there are 1067 images.
#The number of Images is perfect for the Kaggle RAM with respect to the number of images dataset have. This is because, if you have seen my other Pix2Pix GAN notebooks then you would have recognized that I generally use 1000 images for training with 256 size and in this data set, that's the default case.

SIZE = 256
horse_images, zebra_images = np.zeros(shape=(len(horse_paths),SIZE,SIZE,3)), np.zeros(shape=(len(horse_paths),SIZE,SIZE,3))
for i,(horse_path, zebra_path) in tqdm(enumerate(zip(horse_paths, zebra_paths)), desc='Loading'):
    
    horse_image = img_to_array(load_img(horse_path))
    horse_image = tfi.resize(tf.cast(horse_image, tf.float32)/255., (SIZE, SIZE))
    
    zebra_image = img_to_array(load_img(zebra_path))
    zebra_image = tfi.resize(tf.cast(zebra_image,tf.float32)/255., (SIZE, SIZE))
    
    # as the data is unpaired so we don't have to worry about, positioning the images.
    
    horse_images[i] = horse_image
    zebra_images[i] = zebra_image


dataset = [horse_images, zebra_images]

def show_image(image, title=None):
    '''
    The function takes in an image and plots it using Matplotlib.
    '''
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

def show_preds(g_AB, g_BA,n_images=1):
    for i in range(n_images):
        
        id = np.random.randint(len(horse_images))
        horse, zebra = horse_images[id], zebra_images[id]
        horse_pred, zebra_pred = g_BA.predict(tf.expand_dims(zebra,axis=0))[0], g_AB.predict(tf.expand_dims(horse,axis=0))[0]
        
        plt.figure(figsize=(10,8))
        
        plt.subplot(1,4,1)
        show_image(horse, title='Original Horse')
        
        plt.subplot(1,4,2)
        show_image(zebra_pred, title='Generated Zebra')
        
        plt.subplot(1,4,3)
        show_image(zebra, title='Original Zebra')
        
        plt.subplot(1,4,4)
        show_image(horse_pred, title='Genrated Horse')
        
        plt.tight_layout()
        plt.show()
        
   
HtoZ_gen = load_model(r"D:/softweb/cycleGAN_horse2zebra/model_new_v1/g_model_AtoB_000006.h5")
ZtoH_gen = load_model(r"D:/softweb/cycleGAN_horse2zebra/model_new_v1/g_model_BtoA_000006.h5")
show_preds(HtoZ_gen, ZtoH_gen, n_images=5)




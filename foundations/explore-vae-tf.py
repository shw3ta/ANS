"""
a first implementation of a vae
directly references tensorflow documentation.
"""

from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL 
import tensorflow as tf 
import tensorflow_probability as tfp 
import time

print("Got to the end of imports.")

# loading MNIST
(train, _), (test, _) = tf.keras.datasets.mnist.load_data()

# pre-processing
def preproc(imageset):
	images = imageset.reshape((images.shape[0], 28, 28, 1))/255.
	return np.where(images > .5, 1.0, 0.0).astype('float32')

train_ims	= preproc(train) # training imageset
test_ims	= preproc(test) # testing imageset

train_size	= 60000
test_size 	= 10000
batch_size	= 32

# shuffling, batching the data
test_data	= (tf.data.Dataset.from_tensor_slices(train_ims).shuffle(train_size).batch(batch_size))
train_data	= (tf.data.Dataset.from_tensor_slices(test_ims).shuffle(test_size).batch(batch_size))

# we make of Sequential API here

class ConvVAE(tf.keras.Model):

	def __init__(self, latent_dim):
		super(ConvVAE, self).__init__()
		self.latent_dim	= latent_dim

		# encoder network
		self.encoder	= tf.keras.Sequential(
			[
			tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
			tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'),
			tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(latent_dim + latent_dim),
			]
			)

		# decoder network
		self.decoder	= tf.keras.Sequential(
			[
			tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
			tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
			tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
			tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
			tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
			tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
			]
			)

		@tf.function
		def sample(self, eps=None):
			if eps is None:
				eps = tf.random.normal(shape=(100, self.latent_dim))
			return self.decode(eps, apply_sigmoid=True)


		def encode(self, x):
			mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
			return mean, logvar

		def reparameterize(self, mean, logvar):
			eps = tf.random.normal(shape=mean.shape)
			retuen eps * tf.exp(logvar * 0.5) + mean

		def decode(self, z, apply_sigmoid=False):
			logits = self.decoder(z)
			if apply_sigmoid:
				probs = tf.sigmoid(logits)
				return probs
			return logits
#------------------------------------------------------------

# optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)

# setting up the loss function
def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(-.5*((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = raxis)

def compute_loss(model, x):
	mean, logvar = model.encode(x)
	x_logit = model.decode(z)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
	logpx_z = -tf.reduce_sum(cross_entropy, axis=[1,2,3])
	logpz = log_normal_pdf(z, 0.0, 0.0)
	logqz_x = log_normal_pdf(z, mean, logvar)

	return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
	""" 
	exec one step, ret loss

	computes loss and gradients, uses gradients to update model params
	"""

	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#------------------------------------------------------------------

# training phase

epochs	= 10
latent_dim	= 2
num_examples_to_generate = 16

rand_vec = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = ConvVAE(latent_dim)

def generate_and_save_ims(model, epoch, test_sample):
	mean, logvar = model.encode(test_sample)
	z = model.reparameterize(mean, logvar)
	predictions = model.sample(z)
	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0], cmap='gray')
		plt.axis('off')
	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	plt.show()

# pick a sample of the test set for generation of output ims
assert batch_size >= num_examples_to_generate
for test_batch in test_data.take(1):
	test_sample = test_batch[0:num_examples_to_generate, :, :, :]


generate_and_save_ims(model, 0, test_sample)

for epoch in range(1, epochs + 1):
	start_time = time.time()
	for train_x in train_data:
		train_step(model, train_x, optimizer)
	end_time = time.time()

	loss = tf.keras.metrics.Mean()
	for test_x in test_data:
		loss(compute_loss(model, test_x))

	elbo = -loss.result()
	display.clear_output(wait=False)

	print(f"Epoch : {epoch}, Test set ELBO : {elbo}, time elapse for current epoch : {end_time - start_time}")

	generate_and_save_ims(model, epoch, test_sample)
	
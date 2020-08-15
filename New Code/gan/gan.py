from utils import Timer, DataLogger

from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np

class GAN(object):

    def __init__(self):
        # Initialise objects to keep track of log data
        self.logdata = DataLogger()
        self.timer = Timer()

        # Some variables regarding the input size
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.epochs = 0

        # Load the training data
        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(
            np.float32)

        # Initalise the Generator and Discriminator
        self.G = None
        self.D = None
        self.DM = None
        self.AM = None

        self.discriminator = self.discriminator_model()
        self.adversarial = self.adversarial_model()
        self.generator = self.init_generator()

    def save_weights(self, genfile=False, disfile=False):
        # Save the weights of the GAN for later use
        if not genfile:
            genfile = "generator_{}.h5".format(len(self.logdata.D_acc))
        if not disfile:
            disfile = "discriminator_{}.h5".format(len(self.logdata.D_acc))
        self.G.save(genfile)
        self.D.save(disfile)
        print("Saved generator weights to {}".format(genfile))
        print("Saved discriminator weights to {}".format(disfile))

    def load_weights(self, genfile="generator_5000.h5", disfile="discriminator_5000.h5",
                     logfile="MNIST-Logdata-5000-Epochs"):
        # This function loads the weights of a previously trained GAN.
        # The default inputs for this function are weights have been trained for 5000 epochs.
        try:
            self.G.load_weights(genfile)
            print("Loaded generator weights from {}".format(genfile))
        except:
            print("Loading generator weights failed")
        try:
            self.D.load_weights(disfile)
            print("Loaded discriminator weights from {}".format(disfile))
        except:
            print("Loading discriminator weights failed")
        self.logdata.load("MNIST-Logdata-5000-Epochs")
        self.epochs = len(self.logdata.D_acc)

    def train(self, train_steps=1000, batch_size=256):
        # Trains the GAN
        # Start timer
        self.timer.reset()
        for i in range(train_steps):
            # increment counter for number of epochs elapsed
            self.epochs += 1

            # Sample Real Data
            images_train = self.x_train[
                           np.random.randint(0, self.x_train.shape[0], size=batch_size),
                           :, :, :]

            # Sample Fake Data
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)

            # Create Discriminator Dataset
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            # Train Discriminator
            d_loss = self.discriminator.train_on_batch(x, y)

            # Create Generator Data
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # Choose labels to be 1.
            # Plugging this into the cross entropy mean the objective for the generator maximises Log(D(G(x)))
            # This is the heuristic game which gives us better gradients early in training
            y = np.ones([batch_size, 1])

            # Train the Generator
            a_loss = self.adversarial.train_on_batch(noise, y)

            # Print the Log Messages
            log_mesg = "%d: [D loss: %f, acc: %f]" % (self.epochs, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            # Save to the log file
            self.logdata.log([d_loss[1], a_loss[1], d_loss[0], a_loss[0]])
        # Log the time
        self.timer.elapsed_time()

    def init_discriminator(self):
        # Defines the discriminator -- this is a convolutional neural network which tries to see if the image is real
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        input_shape = (28, 28, 1)
        self.D.add(
            Conv2D(depth * 1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        # Sigmoid output ensures that the output is a probability
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def init_generator(self):
        # Defines the discriminator -- this is a deconvolutional neural network which generates images from a prior
        # The prior is a random vector with components uniformly distributed on [-1,1]
        self.G = Sequential()
        dropout = 0.4
        depth = 64 + 64 + 64 + 64
        dim = 7

        self.G.add(Dense(dim * dim * depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Output is 28 x 28 x 1 grayscale image [0.0,1.0] per pixel
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        # Defines the optimiser objective for the discriminator
        # We use the adam gradient method for both networks
        optimizer = Adam(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.init_discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, \
                        metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        # Defines the objective for the generator
        # We use the adam gradient method for both networks
        optimizer = Adam(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.init_generator())
        self.AM.add(self.init_discriminator())
        # Note that we nest the generator and dicriminator -- this means the objective is defined in terms of the discriminator
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, \
                        metrics=['accuracy'])
        return self.AM


if __name__ == "__main__":
    import sys
    print(sys.executable)
    mnist_dcgan = GAN()
    mnist_dcgan.train()
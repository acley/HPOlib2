import time
import numpy as np

import tensorflow as tf
import keras

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import CIFAR10Data


class ConvolutionalNeuralNetwork(AbstractBenchmark):
    def __init__(self, path=None, max_num_epochs=40, rng=None):

        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)
        self.max_num_epochs = max_num_epochs
        self.num_classes = len(np.unique(self.train_targets))

        # setting a random seed in TF is not straightforward
        # see https://github.com/fchollet/keras/issues/2280
        # if rng is None:
        #     self.rng = np.random.RandomState()
        # else:
        #     self.rng = rng
        # lasagne.random.set_rng(self.rng)
        self.rng = rng

        super(ConvolutionalNeuralNetwork, self).__init__()

    def get_data(self, path):
            pass

    def load_mnist_data(self):
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        val_size = 5000
        val_selections = np.random.choice(x_train.shape[0], size=val_size, replace=False)
        val_idx = np.zeros(x_train.shape[0])
        val_idx[val_selections] = 1

        x_val = x_train[val_idx == True]
        y_val = y_train[val_idx == True]
        x_train = x_train[val_idx == False]
        y_train = y_train[val_idx == False]

        return x_train, y_train, x_val, y_val, x_test, y_test

    @AbstractBenchmark._check_configuration
    # @AbstractBenchmark._configuration_as_array
    def objective_function(self, params, steps=1, dataset_fraction=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        # Shuffle training data
        # shuffle = self.rng.permutation(self.train.shape[0])
        shuffle = np.random.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
                                                                      self.valid, self.valid_targets,
                                                                      batch_size=params['batch_size'],
                                                                      num_epochs=num_epochs)

        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve": lc_curve,
                "learning_curve_cost": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, steps=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
                                                                      self.test, self.test_targets,
                                                                      batch_size=params['batch_size'],
                                                                      num_epochs=num_epochs)
        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve": lc_curve,
                "learning_curve_cost": cost_curve}

    @staticmethod
    def get_configuration_space():
        max_layers = 18
        width_choices = [1, 3, 5, 7]
        height_choices = [1, 3, 5, 7]
        N_filt_choices = [24, 36, 48, 64]
        stride_choices = [1, 2, 3] # not used


        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            name='num_layers', lower=1, upper=max_layers, default=7))

        for i in range(max_layers):
            lstr = 'Layer ' + str(i+1)
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                name=lstr + ' Width', choices=width_choices, default=5))
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                name=lstr + ' Height', choices=height_choices, default=5))
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                name=lstr + ' NumFilts', choices=N_filt_choices, default=36))

            for j in range(i):
                name = lstr + ' InputFromLayer ' + str(j+1)
                cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                name=name, choices=[0,1], default=0))
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Convolutional Neural Network',
                'bounds': [[-6, 0],  # init_learning_rate
                           [32, 512],  # batch_size
                           [4, 10],  # n_units_1
                           [4, 10],  # n_units_2
                           [4, 10]],  # n_units_3
                'references': ["@article{klein-bnn16a,"
                               "author = {A. Klein and S. Falkner and T. Springenberg and F. Hutter},"
                               "title = {Bayesian Neural Network for Predicting Learning Curves},"
                               "booktitle = {NIPS 2016 Bayesian Neural Network Workshop},"
                               "month = dec,"
                               "year = {2016}}"]
                }

    def calc_paddings(tensors, verbose=False):
        '''
            Takes a list of 2D convolutional layers and
            calculates symmetric paddings to match the dimensions
            of each layer to the largest layer.

            Returns:
                paddings: a list of paddings that match the syntax
                            of the tf.pad function such that:
                            padded_layers = [tf.pad(tensors[i],
                                                    paddings[i])
                                             for i in range(len(tensors))

        '''

        # extract filter dims without batch_size and num_filter_banks
        dims = np.array([t.get_shape().as_list() for t in tensors])[:,1:-1]
        max_dims = np.max(dims, axis=0)
        required_padding = (dims - max_dims)*-1

        # calculate symmetric paddings
        # paddings before: # of zeros added in front of values of the given dimension
        # paddings after: # of zeros added after the values of the given dimension
        # if total number of needed padding is odd, the left-over zero is added to the beginning
        paddings_before = np.array(required_padding/2, dtype=int)
        odd_dims = (required_padding % 2) != 0
        paddings_before[odd_dims] += 1
        paddings_after = np.array(required_padding/2, dtype=int)

        # reshape/format to match the tf.pad syntax:
        paddings = np.zeros((len(tensors),4,2))
        paddings[:,1:3,:] = [np.stack([paddings_before[i,:],
                                    paddings_after[i,:]],
                                    axis=1) for i in range(len(tensors))]

        if verbose:
            print('layer dims: [tensor x dims]\n', dims)
            print('missing dims:\n', required_padding)
            print('paddings before:\n', paddings_before)
            print('paddings after:\n', paddings_after,)
            print('\n final paddings:',)
            [print('layer {}: \n {}'.format(i, paddings[i]))
             for i in range(len(tensors))]

        return paddings

    def concat_2d_conv_layers(layers, verbose=False):
        '''
            Concatenates a list of 2D convolutional layers. If the dimensions
            are not all equal, symmetric paddings are added to match the
            largest dimensions among the layers.

            Returns:
                conc_layer: a 4D tensor
        '''
        paddings = calc_paddings(layers, verbose=verbose)
        padded_layers = [tf.pad(layers[i], paddings[i]) for i in range(len(layers))]
        conc_layer = keras.layers.concatenate(padded_layers, axis=-1)
        return conc_layer


    def train_net(self, train, train_targets,
                  valid, valid_targets, params,
                  num_epochs=100, batch_size=128):


        layers = build_network(params)

        model = keras.models.Model(inputs=layers[0], outputs=layers[-1])
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])


        tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                  write_graph=True, write_images=False)

        hist = model.fit(train, train_targets,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=2,
                        validation_data=(valid, valid_targets),
                        callbacks=[tensorboard])


        learning_curve = 1 - hist.history['val_acc']
        cost = np.zeros(learning_curve.shape[0])
        train_loss = hist.history['loss']
        valid_loss = hist.history['val_loss']

        # learning_curve[e] = 1 - val_acc / val_batches
        # cost[e] = time.time() - start_time
        # train_loss[e] = train_err / train_batches
        # valid_loss[e] = val_err / val_batches

        return learning_curve, cost, train_loss, valid_loss


class ConvolutionalNeuralNetworkOnMNIST(ConvolutionalNeuralNetwork):

    def get_data(self):
        dm = CIFAR10Data()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = ConvolutionalNeuralNetwork.get_meta_information()
        d["references"].append("@Techreport{krizhevsky-tech09a,"
                               "author = {A. Krizhevsky},"
                               "title = {Learning multiple layers of features from tiny images},"
                               "institution = {University of Toronto},"
                               "year = {2009},"
                               "keywords = {ML}}"
                               )
        return d

import time
import numpy as np
import scipy as sp

import tensorflow as tf
import keras
from keras.regularizers import l2

import ConfigSpace
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import CIFAR10Data
from hpolib.util.data_manager import CIFAR10DataZCAWhitened



class ConvolutionalNeuralNetworkArchSearch(AbstractBenchmark):
    def __init__(self, path=None, max_num_epochs=40, batch_size=128, rng=None):

        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data()
        self.max_num_epochs = max_num_epochs
        self.num_classes = len(np.unique(self.train_targets))
        self.batch_size = batch_size

        # setting a random seed in TF is not straightforward
        # see https://github.com/fchollet/keras/issues/2280
        # if rng is None:
        #     self.rng = np.random.RandomState()
        # else:
        #     self.rng = rng
        # lasagne.random.set_rng(self.rng)
        self.rng = rng

        super(ConvolutionalNeuralNetworkArchSearch, self).__init__()

    def get_data(self, path):
            pass

    @AbstractBenchmark._check_configuration
    # @AbstractBenchmark._configuration_as_array
    def objective_function(self, config, steps=1, dataset_fraction=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        # Shuffle training data
        # shuffle = self.rng.permutation(self.train.shape[0])
        shuffle = np.random.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        params = config.get_dictionary()
        #lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
        hist = self.train_net_logged(train, train_targets, self.valid,
                self.valid_targets, batch_size=self.batch_size,
                params=params, num_epochs=num_epochs)
                                                                      #self.valid, self.valid_targets,
                                                                      #batch_size=self.batch_size,
                                                                      #params=params,
                                                                      #num_epochs=num_epochs)

        #y = lc_curve[-1]
        #c = cost_curve[-1]
        #return {'function_value': y,
        #        "cost": c,
        #        "train_loss": train_loss,
        #        "valid_loss": valid_loss,
        #        "learning_curve": lc_curve,
        #        "learning_curve_cost": cost_curve}
        return hist

    @AbstractBenchmark._check_configuration
    # @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, config, steps=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        params = config.get_dictionary()
        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
                                                                      self.test, self.test_targets,
                                                                      batch_size=self.batch_size,
                                                                      params=params,
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
        return {'name': 'Convolutional Neural Network architecture Search',
                'references': ["@article{zoph2016neural",
                                "title={Neural architecture search with reinforcement learning}",
                                "author={Zoph, Barret and Le, Quoc V}",
                                "journal={arXiv preprint arXiv:1611.01578}",
                                "year={2016}"]
                }

    def build_network(self, params):
        n_layers = params['num_layers']

        input_image = keras.layers.Input(shape=self.image_dim)
        net_layers = [input_image]

        dead_end_layers = np.ones(n_layers-1)
        for ilayer_idx in range(n_layers):
            lstr = 'Layer ' + str(ilayer_idx+1)

            # convolution params
            W = params[lstr + ' Width']
            H = params[lstr + ' Height']
            N = params[lstr + ' NumFilts']

            # extract layer inputs
            ilayer_input_idcs = []
            for jlayer_idx in range(ilayer_idx):
                is_input = params[lstr + ' InputFromLayer ' + str(jlayer_idx+1)]
                if is_input == 1:
                    ilayer_input_idcs.append(jlayer_idx+1)
                    dead_end_layers[jlayer_idx] = 0
            # if a current layer doesn't have an input, use the image
            # as input
            if len(ilayer_input_idcs) == 0:
                ilayer_input_idcs.append(0)
            # add all dead-end layers to the input
            # of the last layer
            if ilayer_idx == (n_layers-1):
                dead_end_layers = np.nonzero(dead_end_layers)[0] + 1
                ilayer_input_idcs.extend(dead_end_layers)

            ilayer_inputs = [net_layers[idx] for idx in ilayer_input_idcs]
            if len(ilayer_inputs) > 1:
                concatenated_input = keras.layers.concatenate(ilayer_inputs, axis=-1)
            else:
                concatenated_input = ilayer_inputs[0]
            # concatenated_input = self.concat_conv_tensors(ilayer_inputs)
            layer_name = 'Layer_' + str(ilayer_idx+1)
            ilayer = keras.layers.Conv2D(filters=N,
                                        kernel_size=(W,H),
                                        padding='same',
                                        use_bias=True,
                                        kernel_regularizer=l2(0.0001),
                                        bias_regularizer=l2(0.0001),
                                        name=layer_name)(concatenated_input)
            ilayer = keras.layers.BatchNormalization(axis=-1)(ilayer)
            ilayer = keras.layers.Activation('relu')(ilayer)
            net_layers.append(ilayer)

        net_layers.append(keras.layers.Flatten()(net_layers[-1]))
        net_layers.append(keras.layers.Dense(10, activation='softmax')(net_layers[-1]))
        return net_layers

    def concat_conv_tensors(self, tensors):
        if len(tensors) == 1:
            return tensors[0]

        tensor_dims = np.array([t.get_shape().as_list() for t in tensors])[:,1:-1]
        max_dims = np.max(tensor_dims, axis=0)
        paddings = (tensor_dims - max_dims)*-1

        padded_tensors = []
        for i,itensor in enumerate(tensors):
            ipadding = paddings[i]
            ipadded_tensor = keras.layers.ZeroPadding2D(padding=[[0,ipadding[0]], [0,ipadding[1]]])(itensor)
            padded_tensors.append(ipadded_tensor)

        concat_tensors = keras.layers.concatenate(padded_tensors, axis=-1)
        return concat_tensors

    def random_crop(self, x, size=(32,32)):
        def crop_img(x, size=(32,32)):
            d1_offset = np.random.randint(8)
            d2_offset = np.random.randint(8)
            return x[d1_offset:d1_offset+size[0], d2_offset:d2_offset+size[1], :]

        cx = np.empty(shape=(x.shape[0], size[0], size[1], x.shape[-1]))
        for i in np.arange(len(x)):
            cx[i] = crop_img(x[i], size=size)

        return cx

    def random_flip(self, x, axis=2):
        flip_idx = np.random.random(x.shape[0]) > 0.5
        x[flip_idx] = np.flip(x[flip_idx], axis=axis)
        return x

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False,
                            random_flip=False, random_crop=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            # self.rng.shuffle(indices)
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            batch_inputs = inputs[excerpt]
            batch_targets = targets[excerpt]
            if random_crop:
                batch_inputs = self.random_crop(batch_inputs.copy())
            if random_flip:
                batch_inputs = self.random_flip(batch_inputs)

            yield batch_inputs, batch_targets

    def iterate_minibatches_endless(self, inputs, targets, batch_size,
            shuffle=True, random_flip=False, random_crop=False):
        assert len(inputs) == len(targets)

        indices = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(indices)
        while True:
            for start_idx in range(0, len(inputs)-batch_size + 1, batch_size):
                excerpt = indices[start_idx:start_idx+batch_size]
                batch_inputs = inputs[excerpt]
                batch_targets = targets[excerpt]
                if random_crop:
                    batch_inputs = self.random_crop(batch_inputs.copy())
                if random_flip:
                    batch_inputs = self.random_flip(batch_inputs.copy())
                yield batch_inputs, batch_targets

    def train_net_no_gen(self, train, train_targets,
                        valid, valid_targets, params,
                        num_epochs=100, batch_size=64):

        layers = self.build_network(params)

        model = keras.models.Model(inputs=layers[0], outputs=layers[-1])
        lr = 0.1
        model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True),
            metrics=['accuracy'])


        # called in every epoch, necessary to use tensorboard
        tb_callback = keras.callbacks.TensorBoard(log_dir='./logs',
                        histogram_freq=1, batch_size=32,
                        write_graph=True, write_grads=True,
                        write_images=True)

        hist = model.fit(x=train, y=train_targets, batch_size=batch_size, epochs=num_epochs,
            verbose=1, validation_data=(valid, valid_targets),
            callbacks=[tb_callback])

        return hist



    def train_net_logged(self, train, train_targets,
                        valid, valid_targets, params,
                        num_epochs=100, batch_size=128):

        layers = self.build_network(params)

        model = keras.models.Model(inputs=layers[0], outputs=layers[-1])
        lr = 0.1
        model.compile(loss=keras.losses.categorical_crossentropy,
            #optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True),
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

        # called in every epoch, necessary to use tensorboard
        tb_callback = keras.callbacks.TensorBoard(log_dir='./logs',
                        histogram_freq=5, batch_size=32,
                        write_graph=True, write_grads=False,
                        write_images=True)

        # called in every epoch, used to update lr
        def paper_lr_schedule(epoch, max_epochs=num_epochs, init_lr=lr):
            if epoch < int(0.5*max_epochs):
                return init_lr
            if (epoch >= int(0.5*max_epochs)) and (epoch < int(0.75*max_epochs)):
                return init_lr/10.
            if epoch >= int(0.75*max_epochs):
                return init_lr/100.
        lr_callback = keras.callbacks.LearningRateScheduler(paper_lr_schedule)

        train_data_generator = self.iterate_minibatches_endless(inputs=train, targets=train_targets,
                            batch_size=batch_size, shuffle=True,
                            random_flip=True, random_crop=True)
        # valid_data_generator = self.iterate_minibatches_endless(inputs=valid, targets=valid_targets,
        #                     batch_size=batch_size, shuffle=False)

        hist = model.fit_generator(generator=train_data_generator,
                            steps_per_epoch=int(len(train)/batch_size),
                            epochs=num_epochs,
                            # validation_data=valid_data_generator,
                            # validation_steps=int(len(valid)/batch_size),
                            validation_data=(valid, valid_targets),
                            callbacks=[tb_callback])#,lr_callback])
        return hist


    def train_net(self, train, train_targets,
                  valid, valid_targets, params,
                  num_epochs=100, batch_size=128):

        start_time = time.time()

        layers = self.build_network(params)

        model = keras.models.Model(inputs=layers[0], outputs=layers[-1])
        # decay_rate = 0.1 / num_epochs
        model.compile(loss=keras.losses.categorical_crossentropy,
            #   optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

        print("Starting training...")

        learning_curve = np.zeros([num_epochs])
        cost = np.zeros([num_epochs])
        train_loss = np.zeros([num_epochs])
        valid_loss = np.zeros([num_epochs])


        for e in range(num_epochs):

            epoch_start_time = time.time()
            train_err = 0
            train_batches = 0


            if (e == int(0.5 * num_epochs)) or (e == int(0.75 * num_epochs)):
                old_lr = keras.backend.get_value(model.optimizer.lr)
                new_lr = 0.1*old_lr
                print('Changing learning rate from {} to {}'.format(
                    old_lr, new_lr))
                keras.backend.set_value(model.optimizer.lr, new_lr)

            for batch in self.iterate_minibatches(train, train_targets, batch_size, shuffle=True):
                inputs, targets = batch
                # random data preprocessing
                inputs = self.random_crop(inputs.copy())
                inputs = self.random_flip(inputs)
                err, acc = model.train_on_batch(inputs, targets)
                train_err += err
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(valid, valid_targets, batch_size, shuffle=False):
                inputs, targets = batch
                err,acc = model.test_on_batch(inputs, targets)

                val_err += err
                val_acc += acc
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(e + 1, num_epochs, time.time() - epoch_start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

            learning_curve[e] = 1 - val_acc / val_batches
            cost[e] = time.time() - start_time
            train_loss[e] = train_err / train_batches
            valid_loss[e] = val_err / val_batches

        return learning_curve, cost, train_loss, valid_loss


class ConvolutionalNeuralNetworkArchSearchOnCIFAR10(ConvolutionalNeuralNetworkArchSearch):

    def get_data(self):
        dm = CIFAR10DataZCAWhitened()
        x_train, y_train, x_val, y_val, x_test, y_test = dm.load()

        self.image_dim = x_train[0].shape

        num_classes = len(np.unique(y_train))
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        print('x_train', x_train.shape)
        print('x_val', x_val.shape)
        print('x_test', x_test.shape)

        x_train = self.pad_images(x_train, padding=(4,4))

        #nt = 200
        #nv = 24
        #x_train = x_train[:nt]
        #x_val = x_val[nt:nt+nv]
        #y_train = y_train[:nt]
        #y_val = y_val[nt:nt+nv]

        return  x_train, y_train, x_val, y_val, x_test, y_test

    def pad_images(self, x, padding=(4,4)):
        return np.pad(x, pad_width=[[0,0], padding, padding, [0,0]],
                      mode='constant', constant_values=0)

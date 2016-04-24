import time
import theano
import lasagne
import numpy as np

import theano.tensor as T

from utils import custom_sgd, iterate_minibatches
from kron_layer import KronLayer
from lowrank_layer import LowRankLayer


def build_custom_mlp(input_var=None, widths=None, drop_input=.2,
                     drop_hidden=.5, type="dense", params=None):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    params = params if params is not None else {}
    widths = widths if widths is not None else [100, 100]
    manifolds = {}

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify

    if type == "dense":
        network = lasagne.layers.DenseLayer(
            network, widths[0], nonlinearity=nonlin)
    elif type == "lowrank":
        param_density = params.get('param_density', 1.0)
        network = LowRankLayer(network, widths[0], param_density=param_density, name="fixedrank0")
        manifolds["fixedrank0"] = network.manifold
    elif type == "kron":
        param_density = params.get('param_density', 1.0)
        shape2 = params.get('shape2', (4, 4))
        network = KronLayer(network, widths[0], shape2=(4, 4), param_density=param_density, name="kron_fixedrank0")
        manifolds["kron_fixedrank0"] = network.manifold
    else:
        raise ValueError("type must be one of 3 variants: 'dense', 'lowrank' or 'kron'")
    for width in widths[1:]:
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network, manifolds


def build_custom_cnn(input_var=None, widths=None, drop_input=.0,
                     drop_hidden=.5, type="dense", params=None):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    params = params if params is not None else {}
    widths = widths if widths is not None else [100, 100]
    manifolds = {}

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify


    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=8, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=8, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    if type == "dense":
        network = lasagne.layers.DenseLayer(
            network, widths[0], nonlinearity=nonlin)
    elif type == "lowrank":
        param_density = params.get('param_density', 1.0)
        network = LowRankLayer(network, widths[0], param_density=param_density, name="fixedrank0")
        manifolds["fixedrank0"] = network.manifold
    elif type == "kron":
        param_density = params.get('param_density', 1.0)
        shape2 = params.get('shape2', (4, 4))
        network = KronLayer(network, widths[0], shape2=(4, 4), param_density=param_density, name="kron_fixedrank0")
        manifolds["kron_fixedrank0"] = network.manifold
    else:
        raise ValueError("type must be one of 3 variants: 'dense', 'lowrank' or 'kron'")
    for width in widths[1:]:
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network, manifolds




def test_nn(X_train,y_train,X_val,y_val,X_test,y_test):
    input_X = T.tensor4("X")
    input_shape = [None,1,28,28]
    target_y = T.vector("target Y integer", dtype='int32')

    #входной слой (вспомогательный)
    input_layer = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X)

    #полносвязный слой, который принимает на вход input layer и имеет 100 нейронов.
    # нелинейная функция - сигмоида как в логистической регрессии
    # слоям тоже можно давать имена, но это необязательно
    units = 48
    dense_1 = KronLayer(input_layer,
                       num_units=units,
                        shape1=(784//4, units//4), shape2=(4, 4),
                        param_density=1.0,
                       name = "fixed_rank")
    """
    dense_1 = lasagne.layers.DenseLayer(input_layer,
                       num_units=units, b=None,
                       name = "dense_1")
    """
    biased_1 = lasagne.layers.BiasLayer(dense_1)
    nonlin_1 = lasagne.layers.NonlinearityLayer(biased_1, lasagne.nonlinearities.rectify)

    #dense_1 = lasagne.layers.DenseLayer(input_layer,num_units=25,
    #                                   nonlinearity = lasagne.nonlinearities.sigmoid,
    #                                   name = "hidden_dense_layer")

    #ВЫХОДНОЙ полносвязный слой, который принимает на вход dense_1 и имеет 10 нейронов -по нейрону на цифру
    #нелинейность - softmax - чтобы вероятности всех цифр давали в сумме 1
    dense_output = lasagne.layers.DenseLayer(nonlin_1,num_units = 50,
                                            nonlinearity = lasagne.nonlinearities.softmax,
                                           name='output')

    #предсказание нейронки (theano-преобразование)
    y_predicted = lasagne.layers.get_output(dense_output)

    #все веса нейронки (shared-переменные)
    all_weights = lasagne.layers.get_all_params(dense_output)
    print(all_weights)

    #функция ошибки - средняя кроссэнтропия
    loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
    l2_penalty = lasagne.regularization.regularize_layer_params(dense_1, lasagne.regularization.l2) * 1e-3
    loss += l2_penalty


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()

    #сразу посчитать словарь обновлённых значений с шагом по градиенту, как раньше
    #updates_sgd = lasagne.updates.sgd(loss, all_weights,learning_rate=0.01)
    updates_sgd = custom_sgd(loss, all_weights, learning_rate=0.01, manifolds={"fixed_rank": dense_1.manifold})

    #функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь и точности
    theano.config.exception_verbosity = 'high'

    train_fun = theano.function([input_X,target_y],[loss,accuracy],updates=updates_sgd)
    accuracy_fun = theano.function([input_X,target_y],accuracy)

    num_epochs = 5 #количество проходов по данным

    batch_size = 50 #размер мини-батча

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,batch_size):
            inputs, targets = batch
            train_err_batch, train_acc_batch= train_fun(inputs, targets)
            train_err += train_err_batch
            train_acc += train_acc_batch
            train_batches += 1

        # And a full pass over the validation data:
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            val_acc += accuracy_fun(inputs, targets)
            val_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
        print("  train accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


def test_nn(X_train,y_train,X_val,y_val,X_test,y_test):
    input_X = T.tensor4("X")
    input_shape = [None,1,28,28]
    target_y = T.vector("target Y integer", dtype='int32')
    dense_output, manifolds = build_custom_cnn(input_X, widths=[100], type="dense", params={'param_density': 0.1})
    #предсказание нейронки (theano-преобразование)
    y_predicted = lasagne.layers.get_output(dense_output)

    #все веса нейронки (shared-переменные)
    all_weights = lasagne.layers.get_all_params(dense_output)
    print(all_weights)

    #функция ошибки - средняя кроссэнтропия
    loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()

    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()

    #сразу посчитать словарь обновлённых значений с шагом по градиенту, как раньше
    #updates_sgd = lasagne.updates.sgd(loss, all_weights,learning_rate=0.01)
    ## TODO don't forget
    updates_sgd = custom_sgd(loss, all_weights, learning_rate=0.01, manifolds=manifolds)

    #функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь и точности
    theano.config.exception_verbosity = 'high'

    train_fun = theano.function([input_X,target_y],[loss,accuracy],updates=updates_sgd)
    accuracy_fun = theano.function([input_X,target_y],accuracy)

    num_epochs = 5 #количество проходов по данным

    batch_size = 50 #размер мини-батча

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,batch_size):
            inputs, targets = batch
            train_err_batch, train_acc_batch= train_fun(inputs, targets)
            train_err += train_err_batch
            train_acc += train_acc_batch
            train_batches += 1

        # And a full pass over the validation data:
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            val_acc += accuracy_fun(inputs, targets)
            val_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
        print("  train accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

if __name__ == "__main__":
    from mnist.mnist import load_dataset
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    print(X_train.shape,y_train.shape)

    #import cProfile
    #cProfile.run("test_nn(X_train,y_train,X_val,y_val,X_test,y_test)")
    test_nn(X_train,y_train,X_val,y_val,X_test,y_test)


import time
import theano
import lasagne
import numpy as np

import theano.tensor as T

from utils import custom_sgd
from kron_layer import KronLayer


def iterate_minibatches(X, y, batchsize):
        n_samples = X.shape[0]

        # Shuffle at the start of epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batchsize):
            end = min(start + batchsize, n_samples)

            batch_idx = indices[start:end]

            yield X[batch_idx], y[batch_idx]
        if n_samples % batchsize != 0:
            batch_idx = indices[n_samples - n_samples % batchsize :]
            yield X[batch_idx], y[batch_idx]


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

if __name__ == "__main__":
    from mnist.mnist import load_dataset
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    print(X_train.shape,y_train.shape)

    #import cProfile
    #cProfile.run("test_nn(X_train,y_train,X_val,y_val,X_test,y_test)")
    test_nn(X_train,y_train,X_val,y_val,X_test,y_test)


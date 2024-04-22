#!/usr/bin/env python3
'''
   Builds, trains, saves NN model
   using Adam optimization, mini-batch gradient descent,
   learning rate decay, and batch normalization in TF
'''

import tensorflow.compat.v1 as tf
import numpy as np


def create_layer(prev, n, activation):
    '''
        Method to create layer
    '''

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    new_layer = tf.layers.Dense(n,
                                activation=activation,
                                kernel_initializer=initializer,
                                name="layer")

    output = new_layer(prev)

    return output


def create_batch_norm_layer(prev, n, activation):
    '''
        Method that creates a batch normalization layer for a
        NN in tf
    '''

    if activation is None:
        return create_layer(prev, n, activation)

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    new_layer = tf.layers.Dense(n,
                                activation=None,
                                kernel_initializer=initializer,
                                name="layer")

    x = new_layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')

    epsilon = 1e-8

    x_norm = tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(x_norm)


def forward_prop(prev, layers, activations, epsilon):
    '''
    '''

    layers_norm = create_batch_norm_layer(prev, layers[0], activations[0])

    for i in range(1, len(layers)):
        layers_norm = create_batch_norm_layer(layers_norm,
                                              layers[i], activations[i])

    return layers_norm


def calculate_accuracy(y, y_pred):
    '''
    '''

    correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                  tf.argmax(y_pred, axis=1))

    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    return accuracy


def calculate_loss(y, y_pred):
    '''
    '''

    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred)

    return loss


def create_train_op(loss, alpha, beta1, beta2, epsilon):
    '''
    '''

    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        name='Adam'
    )

    train = optimizer.minimize(loss)

    return train


def shuffle_data(X, Y):
    '''
    '''
    m = X.shape[0]
    permutted_index = np.random.permutation(m)
    X_shuffled = X[permutted_index]
    Y_shuffled = Y[permutted_index]
    return X_shuffled, Y_shuffled


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    '''
    '''

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, nx = X_train.shape

    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classes))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(initial_value=0,
                              trainable=False,
                              dtype=tf.int32,
                              name='global_step')

    alpha = tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=1,
        global_step=global_step,
        staircase=True)

    train_op = create_train_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:

                sess.run(global_step.assign(i))


                sess.run(alpha)

                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                nbr_batch = m // batch_size + (m % batch_size != 0)

                for j in range(nbr_batch):
                    start = j * batch_size
                    stop = min(start + batch_size, m)

                    X_batch = X_shuffled[start:stop]
                    Y_batch = Y_shuffled[start:stop]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if j > 0 and (j + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )

                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path

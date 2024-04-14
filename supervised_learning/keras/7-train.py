#!/usr/bin/env python3
'''
    Train with Learning Rate Decay
'''

import tensorflow.keras as K

def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    '''
    Function that trains a model using mini-batch gradient descent
    '''
    callback = []
    if early_stopping is True and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)
        callback.append(early_stop)

    if learning_rate_decay and validation_data:
        def scheduler(epochs):
            lr = alpha / (1 + decay_rate * epochs)
            return lr

        inv_time_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1)

        callback.append(inv_time_decay)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)

    return history

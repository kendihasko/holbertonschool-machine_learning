#!/usr/bin/env python3
"""
Vanilla autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dim, hidden_layers, latent_dim):
    """
    Makes a vanilla autoencoder
    """
    encoder_input = keras.layers.Input(shape=(input_dim,))
    encoder_output = encoder_input

    for units in hidden_layers:
        encoder_output = keras.layers.Dense(
            units, activation='relu'
        )(encoder_output)

    latent_space = keras.layers.Dense(
        latent_dim, activation='relu'
    )(encoder_output)
    encoder = keras.models.Model(encoder_input, latent_space)

    decoder_input = keras.layers.Input(shape=(latent_dim,))
    decoder_output = decoder_input

    for units in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(
            units, activation='relu'
        )(decoder_output)

    decoder_output = keras.layers.Dense(
        input_dim, activation='sigmoid'
    )(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_out = encoder(encoder_input)
    auto_out = decoder(auto_out)
    autoenc = keras.models.Model(encoder_input, auto_out)
    autoenc.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoenc

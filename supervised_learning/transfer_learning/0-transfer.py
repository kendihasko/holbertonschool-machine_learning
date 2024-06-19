import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Function to preprocess CIFAR-10 data
def preprocess_data(X, Y):
    X_p = tf.keras.applications.mobilenet_v2.preprocess_input(X)
    Y_p = tf.keras.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Define MobileNetV2 model with frozen layers
base_model = MobileNetV2(include_top=False, input_shape=(32, 32, 3))
for layer in base_model.layers:
    layer.trainable = False

# Build your own model on top of base_model
x = Lambda(lambda image: tf.image.resize(image, (96, 96)))(base_model.output)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks (e.g., to save the best model during training)
checkpoint = ModelCheckpoint('cifar10.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# Train the model
model.fit(x_train, y_train, 
          batch_size=32, 
          epochs=20, 
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])

# Evaluate the model
_, val_accuracy = model.evaluate(x_test, y_test)
print(f"Validation accuracy: {val_accuracy:.4f}")

from tensorflow.keras import layers, models
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from numba import cuda
from keras.models import load_model, save_model
import gc

import tarfile

from mlp import generate_mlp
import numpy as np

from model_factory_bird import train_test_split, parse_examples, preprocess

##################### Loading Data #####################
classes = 3
batch_size = 4
epochs = 12
height, width = 256, 256
image_tup = (height, width)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]) 

raw_dataset_train= tf.data.TFRecordDataset(['birds-vs-squirrels-train.tfrecords'])
raw_dataset_valid = tf.data.TFRecordDataset(['birds-vs-squirrels-validation.tfrecords'])

train_base, test_base = train_test_split(raw_dataset_train, 0.8)

train_base = train_base.map(parse_examples, num_parallel_calls=16)
test_base = test_base.map(parse_examples, num_parallel_calls=16)

val_base = raw_dataset_valid.map(parse_examples, num_parallel_calls=16)
val_X = []
val_y = []
for image, label in val_base:
    val_X.append(image.numpy())  # Convert to numpy array
    val_y.append(label.numpy())
val_x = np.array(val_X)
val_y = np.array(val_y)

train = train_base.map(preprocess, num_parallel_calls=16)
test = test_base.map(preprocess, num_parallel_calls=16)
valid = val_base.map(preprocess, num_parallel_calls=16)

train = train.batch(batch_size)
test = test.batch(batch_size)
valid = valid.batch(batch_size)

input_shape = (height, width, 3)
inputs = Input(shape=input_shape)


##################### Loading Model #####################
num_layers = 5
init_neurons = 41
scaling_factor = .72
dropout_rate = .347
learning_rate = 0.005735017949362546
activation = 'sigmoid'

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5, 
    min_delta=0.01,  
    restore_best_weights=True  
)
v2l = tf.keras.applications.EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, classes),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
        include_preprocessing=True,
    )
base_model = v2l

base_model.trainable = False
x = base_model(inputs, training=False) # Inputs is defined in section above. 
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Define your custom top layer
top_layer = generate_mlp(num_layers=num_layers, 
                            initial_neurons=init_neurons, 
                            output_classes=classes, 
                            dropout_rate=dropout_rate, 
                            activation_function=activation,
                            scaling_factor=scaling_factor)

x = top_layer(x)

model = Model(inputs, x)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, epochs=epochs, batch_size = batch_size, validation_data=test, verbose=1, callbacks=[early_stopping])
del train, test, train_base, test_base
acc = model.evaluate(val_x, val_y, verbose=0) 
print("Validation Accuracy:", acc)

save_model(model, 'birdsVsSquirrelsModel.h5')

with tarfile.open('birdsVsSquirrelsModel.tar.gz', 'w:gz') as tar:
    tar.add('birdsVsSquirrelsModel.h5', arcname='birdsVsSquirrelsModel.h5')

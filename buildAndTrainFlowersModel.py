##################### Modules #####################
# Internal Pacakges
import gc
import logging
import tarfile

# Repository Packages.-
from preprocessDefinition import preprocess, generate_mlp, train_test_split, parse_examples_flower

# External Modules
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from numba import cuda
from keras.models import load_model, save_model
import numpy as np


##################### Config #####################
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logging.info("Finished Imports")

classes = 102
batch_size = 5
epochs = 25
height, width = 256, 256
image_tup = (height, width)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration]) 

##################### Loading Data #####################
logging.info("Loading in Data")
dataset_name = "oxford_flowers102"
raw_dataset_train, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

train_base, test_base = train_test_split(raw_dataset_train, 0.8)

train_base = train_base.map(parse_examples_flower, num_parallel_calls=16)
test_base = test_base.map(parse_examples_flower, num_parallel_calls=16)

train = train_base.map(preprocess, num_parallel_calls=16)
test = test_base.map(preprocess, num_parallel_calls=16)

train = train.batch(batch_size)
test = test.batch(batch_size)

input_shape = (height, width, 3)
inputs = Input(shape=input_shape)


##################### Loading Model #####################
logging.info("Loading in Model")
num_layers = 10
init_neurons = 600
scaling_factor = .75641
dropout_rate = .04489
learning_rate = 0.009605915177560863
activation = 'tanh'

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5, 
    min_delta=0.01,  
    restore_best_weights=True  
)
logging.info("Loading in Model")
cxl = tf.keras.applications.ConvNeXtXLarge(
        model_name="convnext_xlarge",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, 3),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )
base_model = cxl

##################### Creating Model #####################
logging.info("Creating Model")
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

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

##################### Fitting Model #####################
logging.info("Training Model")
model.fit(train, epochs=epochs, batch_size = batch_size, validation_data=test, verbose=1, callbacks=[early_stopping])

##################### Model Saving #####################
logging.info("Saving Model")

model.save('flowersModel')
with tarfile.open(f'flowersModel.tgz', 'w:gz') as tar:
    tar.add('flowersModel')

logging.info("Model Saved & Script Complete")

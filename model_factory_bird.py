import pandas as pd
import numpy as np
import tensorflow as tf
from mlp import generate_mlp
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# Bayesian Optimization Libraries for hyperparameter tuning.
import optuna
from numba import cuda
import gc

from tensorflow.keras import layers, models

height, width = 256, 256
image_tup = (height, width)
classes = 3
epochs = 15
hyp_calls = 30
batch_size = 4
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]) 




def train_test_split(dataset, train_size_ratio=0.7, shuffle_buffer_size=500):

    dataset_size = sum(1 for _ in dataset)

    train_size = int(train_size_ratio * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset

def parse_examples(serialized_examples):
    feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                         'label':tf.io.FixedLenFeature([],tf.int64)}
    examples=tf.io.parse_example(serialized_examples, feature_description)
    labels=examples.pop('label')
    labels = tf.one_hot(labels, depth=classes) 
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'],channels=3),tf.float32),299,299)
    return images, labels

def preprocess(image, label):

    image = tf.image.resize(image, image_tup)
    tf.keras.applications.efficientnet_v2.preprocess_input(image)

    return image, label

def load_data():
    raw_dataset_train= tf.data.TFRecordDataset(['birds-vs-squirrels-train.tfrecords'])
    raw_dataset_valid = tf.data.TFRecordDataset(['birds-vs-squirrels-validation.tfrecords'])
    
    train_base, test_base = train_test_split(raw_dataset_train, 0.8, shuffle_buffer_size=10000)
    
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

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5, 
        min_delta=0.01,  
        restore_best_weights=True  
    )
   
    
    train = train_base.map(preprocess, num_parallel_calls=16)
    test = test_base.map(preprocess, num_parallel_calls=16)
    valid = val_base.map(preprocess, num_parallel_calls=16)
    
    train = train.batch(batch_size)
    test = test.batch(batch_size)
    valid = valid.batch(batch_size)
    
    input_shape = (height, width, 3)
    inputs = Input(shape=input_shape)

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

    r50 = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, classes),
        pooling=None,
        classes=3,
        )

    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, classes),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
        )
    cxl = tf.keras.applications.ConvNeXtXLarge(
        model_name="convnext_xlarge",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, classes),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )

    v2m = tf.keras.applications.EfficientNetV2L(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(height, width, classes),
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    return train, test, valid, val_x, val_y, v2l, vgg16, v2m, r50, cxl, inputs, early_stopping
train, test, valid, val_x, val_y, v2l, vgg16, v2m, r50, cxl, inputs, early_stopping = load_data() 



def get_model_from_str(base_model_str, v2l, vgg16, v2m, r50, cxl):
    if base_model_str == 'v2l':
        return v2l
    elif base_model_str == 'vgg16':
        return vgg16
    elif base_model_str == 'v2m':
        return v2m
    elif base_model_str == 'r50':
        return r50
    elif base_model_str == 'cxl':
        return cxl
    else:
        raise ValueError(f"Unknown model string: {base_model_str}")

def create_model(optimizer:str, base_model:str, activation, init_neurons, num_layers, scaling_factor, dropout_rate, learning_rate):
    
    base_model = get_model_from_str(base_model, v2l, vgg16, v2m, r50, cxl)
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

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    
def objective(trial):
    # Optuna suggests the parameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    base_model = trial.suggest_categorical('base_model', ['r50', 'v2l', 'vgg16', 'v2m',  'cxl'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    init_neurons = trial.suggest_int('init_neurons', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 10)
    scaling_factor = trial.suggest_float('scaling_factor', 0.01, 0.99)
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.99)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)

    print(f"Suggested parameters for trial {trial.number}:")
    print(f"optimizer: {optimizer}")
    print(f"base_model: {base_model}")
    print(f"activation: {activation}")
    print(f"init_neurons: {init_neurons}")
    print(f"num_layers: {num_layers}")
    print(f"scaling_factor: {scaling_factor}")
    print(f"dropout_rate: {dropout_rate}")
    print(f"learning_rate: {learning_rate}")

    

    model = create_model(optimizer, base_model, activation, init_neurons, num_layers, scaling_factor, dropout_rate, learning_rate)
    
    
    
    # Assume you have training data (X_train, y_train)
    model.fit(train, epochs=epochs, batch_size = batch_size, validation_data=test, verbose=1, callbacks=[early_stopping])
    
    acc = model.evaluate(val_x, val_y, verbose0) 
    del model

    return -acc[1]
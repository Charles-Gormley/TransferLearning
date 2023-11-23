##################### Modules #####################
# Internal Pacakges
import gc
import logging

# Repository Packages.-
from mlp import generate_mlp
from preprocessDefinition import preprocess

# External Packages
import tensorflow as tf
import tensorflow_datasets as tfds
import optuna
from numba import cuda
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten


##################### Hyperparams #####################
height, width = 256, 256
image_tup = (height, width)
classes = 102
epochs = 25
hyp_calls = 30
batch_size = 4
gpus = tf.config.experimental.list_physical_devices('GPU')


##################### Model Definition #####################
v2l = tf.keras.applications.EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(height, width, 3),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
        include_preprocessing=True,
    )

r50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(height, width, 3),
    pooling=None,
    classes=3,
)

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


##################### Preprocessing #####################
def preprocess(image, label):
    image = tf.image.resize(image, image_tup)
    tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

def parse_examples(example):
    # The 'image' key in the dataset will contain the actual image content
    image = example['image']
    
    # Resize the image to your desired size (299x299 in your original function)
    image = tf.image.resize_with_pad(image, 299, 299)

    # Convert the image data to float32
    image = tf.cast(image, tf.float32)

    # The 'label' key contains the class labels
    label = example['label']

    # Convert labels to one-hot encoding
    label = tf.one_hot(label, depth=102)

    return image, label

##################### Loading Dataset #####################
def train_test_split(dataset, train_size_ratio=0.7, shuffle_buffer_size=500):

    dataset_size = sum(1 for _ in dataset)

    train_size = int(train_size_ratio * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset


def load_data():
    dataset_name = "oxford_flowers102"
    raw_dataset_train, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)
    
    
    train_base, test_base = train_test_split(raw_dataset_train, 0.8, shuffle_buffer_size=10000)
    
    train_base = train_base.map(parse_examples, num_parallel_calls=16)
    test_base = test_base.map(parse_examples, num_parallel_calls=16)
    

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5, 
        min_delta=0.01,  
        restore_best_weights=True  
    )
   
    
    train = train_base.map(preprocess, num_parallel_calls=16)
    test = test_base.map(preprocess, num_parallel_calls=16)
    
    
    train = train.batch(batch_size)
    test = test.batch(batch_size)
    
    
    input_shape = (height, width, 3)
    inputs = Input(shape=input_shape)
    

    return train, test, inputs, early_stopping

##################### Model Creation #####################
def get_model_from_str(base_model_str, v2l, r50, cxl):
    if base_model_str == 'v2l':
        return v2l
    elif base_model_str == 'r50':
        return r50
    elif base_model_str == 'cxl':
        return cxl
    else:
        raise ValueError(f"Unknown model string: {base_model_str}")


def create_model(optimizer:str, base_model:str, activation, init_neurons, num_layers, scaling_factor, dropout_rate, learning_rate, inputs):
    
    base_model = get_model_from_str(base_model, v2l, r50, cxl)
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


##################### Hyperparameter Tuning Setup #####################
def objective(trial):
    # Optuna suggests the parameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    base_model = trial.suggest_categorical('base_model', ['r50', 'v2l', 'cxl'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    init_neurons = trial.suggest_int('init_neurons', 32, 1028)
    num_layers = trial.suggest_int('num_layers', 1, 10)
    scaling_factor = trial.suggest_float('scaling_factor', 0.01, 0.99)
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.99)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)

    train, test, inputs, early_stopping = load_data() 

    model = create_model(optimizer, base_model, activation, init_neurons, num_layers, scaling_factor, dropout_rate, learning_rate, inputs)
    
    # Assume you have training data (X_train, y_train)
    history = model.fit(train, epochs=epochs, batch_size = batch_size, validation_data=test, verbose=1, callbacks=[early_stopping])
    test_accuracy = history.history['val_accuracy']
    
    
    model_parameters = (
        f"Optimizer: {optimizer}\n"
        f"Base Model: {base_model}\n"
        f"Activation Function: {activation}\n"
        f"Initial Neurons: {init_neurons}\n"
        f"Number of Layers: {num_layers}\n"
        f"Scaling Factor: {scaling_factor}\n"
        f"Dropout Rate: {dropout_rate}\n"
        f"Learning Rate: {learning_rate}\n"
    )

    print("Model Parameters:\n", model_parameters)

    
    return -test_accuracy[-1]# Last val accuracy.-


if __name__ == "__main__":
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hyp_calls, gc_after_trial=True)
    # Get the best parameters
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Best objective value achieved
    best_value = study.best_value
    print("Best Objective Value:", best_value)
    for trial in study.trials:
        print("Trial Number:", trial.number)
        print("Params:", trial.params)
        print("Value:", trial.value)

    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)
    plot_contour(study)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import logging

def preprocess(image, label):
    height, width = 256, 256
    image_tup = (height, width)


    image = tf.image.resize(image, image_tup)
    tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

def parse_examples_bird(serialized_examples):
    classes = 3
    feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                         'label':tf.io.FixedLenFeature([],tf.int64)}
    examples=tf.io.parse_example(serialized_examples, feature_description)
    labels=examples.pop('label')
    labels = tf.one_hot(labels, depth=classes) 
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'],channels=3),tf.float32),299,299)
    return images, labels

def parse_examples_flower(example):
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

def train_test_split(dataset, train_size_ratio=0.7, shuffle_buffer_size=500):

    dataset_size = sum(1 for _ in dataset)

    train_size = int(train_size_ratio * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset

def generate_mlp(num_layers:int, initial_neurons:int, output_classes:int, dropout_rate:float, activation_function:str, scaling_factor:float):
    '''scaling factor - between 0 and 1.'''
    model = tf.keras.Sequential()

    for i in range(num_layers):
        # For the first layer, add the specified number of neurons
        if i == 0:
            model.add(Dense(initial_neurons, activation=activation_function))
        else:
            # Decrease the number of neurons by the scaling factor
            neurons = int(initial_neurons * (scaling_factor ** i))
            model.add(Dense(neurons, activation=activation_function))

        # Add dropout after each Dense layer
        model.add(Dropout(dropout_rate))

    # Add the final layer with output_classes neurons
    model.add(Dense(output_classes, activation='softmax'))
    logging.debug("MLP Layer Completed.")


    return model
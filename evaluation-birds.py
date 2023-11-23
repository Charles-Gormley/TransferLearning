# External Modules
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



# Repository Modules
from preprocessDefinition import parse_examples_bird, preprocess

##################### Model Params #####################
batch_size = 5

# Load your birds vs squirrels model
model = load_model('birdsVsSquirrelsModel')

##################### Preprocessing #####################
raw_dataset_valid = tf.data.TFRecordDataset(['birds-vs-squirrels-validation.tfrecords'])

# Map the dataset through the parsing and preprocessing functions
val_base = raw_dataset_valid.map(parse_examples_bird, num_parallel_calls=16)
valid = val_base.map(preprocess, num_parallel_calls=16)
valid = valid.batch(batch_size)

##################### Model Evaluation #####################
# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(valid, verbose=1)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

##################### Generate Visuals #####################

predictions = model.predict(valid)
predictions = np.argmax(predictions, axis=1)
# Extract true labels
true_labels = np.concatenate([y for _, y in valid], axis=0)

if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:
    true_labels = np.argmax(true_labels, axis=1)

# Now compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=False, fmt='.2%', cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('birds_confusion.png')
plt.close()
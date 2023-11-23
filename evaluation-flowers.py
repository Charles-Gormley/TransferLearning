# External Modules
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Repository Modules
from preprocessDefinition import preprocess

##################### Model Params #####################

optimizer = tf.keras.optimizers.SGD()

#load your flower model
model=tf.keras.models.load_model('flowersModel')

##################### Preprocessing #####################
evalset, info = tfds.load(name='oxford_flowers102', split='test',as_supervised=True, with_info=True)
evalData = evalset.map(preprocess, num_parallel_calls=16).batch(32).prefetch(1)

top2err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2,name='top2')
top5err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name='top5')
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy',top2err,top5err])

##################### Evaluating Model #####################
val_loss, val_accuracy, val_top2, val_top5 = model.evaluate(evalData)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Top 2 Accuracy: {val_top2}")
print(f"Top 5 Accuracy: {val_top5}")

##################### Generate Visualizations #####################
true_labels = np.concatenate([y for x, y in evalData], axis=0)
predictions = model.predict(evalData)
predictions = np.argmax(predictions, axis=1)


conf_matrix = confusion_matrix(true_labels, predictions, labels=range(info.features['label'].num_classes))

# Normalizing by row (by true labels)
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Creating the heatmap with normalized data
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percentage, annot=False, fmt='.2%', xticklabels=False, yticklabels=False) # fmt can be adjusted as needed
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Percentage')
plt.savefig('flowers_confusion_percentage.png')
plt.close()  # Closes the plot window
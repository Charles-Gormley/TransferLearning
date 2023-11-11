import tensorflow as tf
from mlp import generate_mlp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class Model:
    def __init__(base_model, optimizer:str = "adam", loss:str = 'categorical_crossentropy', custom_top_layer:bool = False, mlp=None):
        self.base_model = base_model

        self.optimizer = optimizer
        self.loss = loss

    def add_top_layer(self, classes:int, input_shape:list = [224, 224, 3]):
        self.base_model.trainable = False
        input = Input(shape=input_shape)
        x = self.base_model(input, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        top_layer = generate_mlp(num_layers=5, initial_neurons=512, output_classes=3, dropout_rate=0.5, activation_function='relu')
        x = top_layer(x)
        model = Model(inputs, x)

        self.model = model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])


if __name__ == "__main__":
    v2l = tf.keras.applications.EfficientNetV2L(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
    )

    v2m = tf.keras.applications.EfficientNetV2L(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
    )

    cxl = tf.keras.applications.ConvNeXtXLarge(
        model_name="convnext_xlarge",
        include_top=True,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    r50 = tf.keras.applications.ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
    )

    vgg16 = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
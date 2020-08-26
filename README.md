# ml-snippets

Um conjunto de anotações sobre coisas comuns para usar em scaffold rápido de modelos.

### ConvNet simples Multi-Categoria: ###

```python
class CustomEarlyStopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["acc"]
        if accuracy >= 0.998:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = [
    CustomEarlyStopCallback()
]

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (5,5), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model fitting
history = model.fit(training_images, training_labels, epochs=20, callbacks=callbacks)

# model fitting
return history.epoch, history.history['acc'][-1]
```

### ImageDataGenerator ###

Dentro do diretório dataset deve existir diretórios para cada rótulo específico.
O Generator atribui o rótulo equivalente ao subdiretório onde o dado se encontra.

Ex:
```
training_data/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

```python
train_datagen = ImageDataGenerator(rescale=1/255) # Your Code Here

# Please use a target_size of 150 X 150.
train_generator = train_datagen.flow_from_directory(
    "./training_data",
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)

history = model.fit_generator(
    # Your Code Here
    train_generator,
    steps_per_epoch=8,  
    epochs=15,
    verbose=1,
    callbacks=[callbacks]
)
# model fitting
    return history.history['acc'][-1]
```

### Keras Quick Reference ###

#### Generators: ####

Os Generators são iteradores turbinados para preprocessamentos de entrada. São capazes de data augmentation, preprocessamento, normalização, split de dataset, etc. Quando usar generators no input, usar `fit_generator` em vez de `fit` no `Model`.

* Image https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

```python
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
```

* Text https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory

```python
tf.keras.preprocessing.text_dataset_from_directory(
    directory, labels='inferred', label_mode='int', class_names=None, batch_size=32,
    max_length=None, shuffle=True, seed=None, validation_split=None, subset=None,
    follow_links=False
)

```

#### Layers: ####

Tipos de camadas mais comuns em modelos do keras.

* Conv2D - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

```python
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs
)
```

* Dense - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    **kwargs
)
```

* Dropout - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

```python
tf.keras.layers.Dropout(
    rate, noise_shape=None, seed=None, **kwargs
)
```


* Flatten - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

```python
tf.keras.layers.Flatten(
    data_format=None, **kwargs
)
```

* MaxPool2D - https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D

```python
tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs
)
```

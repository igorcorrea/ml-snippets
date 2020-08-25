# ml-snippets

### ConvNet simples (multi-categoria): ###

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

### Keras Quick Reference ###

#### Layers: ####

* Conv2D - [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)

```python
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs
)
```

* Dense - [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    **kwargs
)
```

* Dropout - [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)

```python
tf.keras.layers.Dropout(
    rate, noise_shape=None, seed=None, **kwargs
)
```


* Flatten - [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)

```python
tf.keras.layers.Flatten(
    data_format=None, **kwargs
)
```

* MaxPool2D - [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)

```python
tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs
)
```

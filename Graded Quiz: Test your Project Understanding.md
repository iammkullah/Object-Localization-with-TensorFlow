## Graded Quiz: Test your Project Understanding

# Question 1
In a typical object detection task, we may have multiple instances of multiple objects in the given images and all of those instances are classified and localized. This is different from the object localization task that we performed in the guided project, where we trained a model with the assumption that there will always be exactly one instance of one of the objects in every input image. Is this true or false?
* <b>True</b>
* False

# Question 2
We created an iterable data generator in the guided project that yields a dictionary for input data and another dictionary for output data each time it is iterated over. The key names in these dictionaries must match a corresponding layer in the CNN model. Is it true or false?
* False
* <b>True</b>

# Question 3
We used a custom metric called IoU or Intersection over Union while training the model. Provided the following two values, what does the metric mean?

area_i = Area of intersection between the predicted bounding box and the ground truth bounding box

area_u = Area of union between the predicted bounding box and the ground truth bounding box


* iou = (area_u - area_i) / area_u
* iou = area_u / area_i
* <b> iou = area_i / area_u </b>

# Question 4
While compiling a multi output model in Keras, is it valid to provide the loss functions as a dictionary? For example, would the following code be valid?
```python
model.compile(
    loss={
        'output_1': 'binary_crossentropy',
        'output_2': 'mae'
    },
    optimizer=tf.keras.optimizers.Adam()
)
```

* No, it's not valid
* <b> Yes, it's valid </b>

# Question 5
In order to use a custom learning rate schedule during model training, we can use the LearningRateScheduler callback in Keras. Which of the following is a correct implementation of a learning rate scheduler callback that reduces the learning rate by half at the end of each epoch?

<b>(a)-</b>
```python
      def lr_schedule(epoch, lr):
          return lr / 2

      lr_callback = LearningRateScheduler(lr_schedule)
```

(b)-
```python
lr_callback = LearningRateScheduler(0.5)
```

# Question 6
Please fill in the blank in order to create and initialize a numpy array of shape (32, 32) with 0 values:

X = np._____((32, 32)

Where np is numpy and you just need to type in the function name

Answer : <b> Zeros </b>

# Question 7
How would you create an Input tensor to be used as input to a functional Keras model? Assume the input shape to be (16, 4)

(a)-

from tensorflow.keras.layers import Input

x = Input(shape=(16, 4))(x)

<b>(b)-</b>

from tensorflow.keras.layers import Input

x = Input(shape=(16, 4))

# Question 8
In order to create a custom metric using Keras, which Keras class would you subclass?

* tensorflow.keras.Metric
* <b> tensorflow.keras.metrics.Metric </b>

# Question 9
You've imported tensorflow as tf. How would you find the maximum value between two tensors T1 and T2? Assume that your code needs to work in graph mode (like with custom metric we created)
* tf.max(T1, T2)
* <b> tf.maximum(T1, T2) </b>
* It is not possible to do this in graph mode


# Question 10
Would it be possible to create the same two output model we created in this project using the Keras sequential API instead?
* <b> No </b>
* Yes



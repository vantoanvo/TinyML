import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import math 

#Create sample size
SAMPLES = 1000
#Set seed value to keep the same random number everytime
SEED = 1337 
np.random.seed(SEED)
tf.random.set_seed(SEED) #if only set np.random.seed(SEED), the results might be different for model training tf

#generate uniformed set of random numbers in the range from 0 to 2n,
#which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

#shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)

#Calculate the corresponding sine values
y_values = np.sin(x_values)

#add some random noise to each y value, use y_values.shape to make sure the new data is the same shape as y_values
y_values += 0.1 * np.random.randn(*y_values.shape)

#split the data: 20% for validation, 20% for testing, 60% for training
TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

#use np.split to chop date into three parts
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT]) #from 0 to TRAIN_SPLIT, from TRAIN_SPLIT to TEST_SPLIT
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

#plot the data in each partition in different colors
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.plot(x_test, y_test, 'r.', label="Test")

#build a regression model
model_1 = tf.keras.Sequential()

#first layer takes a scalar input and feeds it through 16 "neurons"
#neurons decide whether to activate based on the 'relu' activation function
#1 means the input is a single value
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))

#final layer is a single neuron, since we want to output a single value
model_1.add(layers.Dense(1))

#compile the model using a standard optimizer and loss function for regression
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

#print the model's architecture
model_1.summary()
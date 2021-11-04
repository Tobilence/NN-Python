from Library.Accuracy import *
from Library.Activations import *
from Library.Layers import *
from Library.Loss import *
from Library.Optimizers import *
from Library.Model import Model
import nnfs
from nnfs.datasets import spiral_data, sine_data

nnfs.init()

# Softmax + Categorical Crossentropy
"""
X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
"""

# Binary Logistic Regression
"""
nnfs.init()
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model()

model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())

model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-7),
    accuracy=Accuracy_Categorical(binary=True)
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
"""

# Regression
"""
nnfs.init()
X, y = sine_data()

model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3), accuracy=Accuracy_Regression())

model.finalize()

model.train(X, y, epochs=10000, print_every=100)
"""




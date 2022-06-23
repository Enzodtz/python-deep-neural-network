# python-deep-neural-network

Deep Neural Network Model with Gradient Descend Optimization applied in python

### Getting Started

To use this model, you only need to:

1. Instantiate the model
2. Train it
3. Predict values with it
4. Save the model
5. Open it later on

```python
from deep_neural_network import *

model = DeepNeuralNetwork()
model.train(X_train, Y_train) # refer to function for more details
model.predict(X)
model.save('filename.pickle')
model = DeepNeuralNetwork.load('filename.picke')
```

### Demo

To see how you can use this model with all its functionalities, refer to `example.py`.

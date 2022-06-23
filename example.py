import numpy as np
from deep_neural_network import DeepNeuralNetwork
from PIL import Image
import h5py
import matplotlib.pyplot as plt


def load_data():
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(
    train_x_orig.shape[0], -1
).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

layers_dims = [12288, 20, 50, 20, 1]

model = DeepNeuralNetwork()
model.train(train_x, train_y, layer_dims=layers_dims, num_iterations=20000)


pred_train = model.predict(train_x)
train_acc = np.sum((pred_train == train_y) / train_x.shape[1])
print("Accuracy Train:", train_acc)
pred_test = model.predict(test_x)
test_acc = np.sum((pred_test == test_y) / test_x.shape[1])
print("Accuracy Test:", test_acc)

model.save("demo.pickle")

# Model stats
plt.figure()
plt.title("Model Costs")
plt.plot(model.costs)
plt.show()

plt.figure()
plt.title("Model Accuracy")
plt.bar(["Train", "Test"], [train_acc, test_acc])
plt.show()

# Predicting with other image, notice that it should be pre-processed to fit the model
image = Image.open("images/cat.jpeg")
img_show = image.resize((num_px, num_px), Image.ANTIALIAS)
image = np.asarray(img_show)
image = image.flatten()
image = image.reshape((image.shape[0], 1))
image = image / 255.0
prediction = model.predict(image)

plt.title("Predicted as: " + ("cat" if prediction else "non-cat"))
plt.imshow(img_show)
plt.show()

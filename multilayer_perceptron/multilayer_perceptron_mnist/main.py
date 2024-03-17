import numpy as np
from keras.datasets import mnist

from multilayer_perceptron_mnist import MultilayerPerceptron
from constants import IMG_MAX_VALUE, SAMPLE_SIZE

# preparing data

train_data, test_data = mnist.load_data()

train_imgs, train_labels = (np.array(d[0:SAMPLE_SIZE]) for d in train_data)
test_imgs, test_labels = (np.array(d[0:SAMPLE_SIZE]) for d in test_data)

train_imgs = np.array([img.flatten() for img in train_imgs])
test_imgs = np.array([img.flatten() for img in test_imgs])

train_imgs = np.array([img.reshape(1, len(img))  for img in train_imgs])
test_imgs = np.array([img.reshape(1, len(img)) for img in test_imgs])

# end of data preparation

model = MultilayerPerceptron()
model.train(train_imgs, train_labels)

count = 0

for img, label in zip(test_imgs, test_labels):
    prediction = model.predict(img)
    if prediction == label:
        count += 1

print(f'Total of right predictions: {count}')
print(f'Precision: {count / len(test_labels) * 100:.2f}%')
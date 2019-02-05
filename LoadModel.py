from keras.datasets import mnist
from keras.models import load_model

loaded_model = load_model('models/model.h5')

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plot the first image in the dataset
# print(plt.imshow(X_train[0]))

#check image shape
# print(X_train[0].shape)

#reshape data to fit model

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

print(loaded_model.predict(X_test[:4]))
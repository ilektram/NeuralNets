import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split

# Load data as separate ndarrays
pole_angle, pole_ang_velocity, pole_expected = np.loadtxt(
    "pole_balancing.txt",
    skiprows=3,
    unpack=True
)

# Convert data to correct dimensions
pole_inp = np.column_stack((pole_angle, pole_ang_velocity))
pole_expected = pole_expected.reshape(len(pole_inp), 1)

# Split data to training & testing sets
X_train, X_test, y_train, y_test = train_test_split(pole_inp,
                                                    pole_expected,
                                                    test_size=0.33,
                                                    random_state=42)

# Create NN for 2-layer unidimensional regression
model = Sequential()
model.add(Dense(2, input_shape=(pole_inp.shape[1],)))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

learning_rate = .1
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd)

epochs = 10
model.fit(X_train, y_train, nb_epoch=epochs, batch_size=200)
# score = model.evaluate(X_test, y_test, batch_size=16)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print("This NN's RMSE: {}".format(rmse))

score = model.evaluate(X_test, y_test, show_accuracy=True, batch_size=200)

print('Test score: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))

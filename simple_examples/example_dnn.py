import numpy as np
from numpy.random import randint, standard_normal
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

#basic problem definitions
num_features = 4 #number of input features (dimension of input vector x)
num_classes = 3 #number of classes. Correct label is y \in {0, 1,..., num_classes}
num_train_examples = 10 # number of training examples

#generate some random data
X_train = standard_normal((num_train_examples, num_features))
y_train = randint(num_classes, size=num_train_examples)
y_train_onehot = to_categorical(y_train) #convert integers to one-hot encoding
X_test = standard_normal((num_train_examples, num_features))
y_test = randint(num_classes, size=num_train_examples)
y_test_onehot = to_categorical(y_test) #convert integers to one-hot encoding

lr=0.01
lr_decay=0.0001
optimizer = Adam(lr=lr, decay=lr_decay)
n_epochs=200
n_mini_batch=60

neural_net = Sequential()
neural_net.add(Dense(100, input_shape=(num_features,)))
neural_net.add(Dropout(0.5))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(num_classes, activation='softmax'))

#training stage
neural_net.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

history = neural_net.fit(
            X_train,
            y_train_onehot,
            batch_size=n_mini_batch,
            epochs=n_epochs,
        )

#test stage
neural_net_outputs = neural_net.predict(X_test)
y_predicted = np.argmax(neural_net_outputs, axis=1)
print('Accuracy = ', accuracy_score(y_test, y_predicted))

print(X_train)
print(y_train)
print(y_train_onehot)
print(y_predicted)
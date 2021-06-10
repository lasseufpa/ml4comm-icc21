from numpy.random import randint, standard_normal
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#basic problem definitions
num_features = 4 #number of input features (dimension of input vector x)
num_classes = 3 #number of classes. Correct label is y \in {0, 1,..., num_classes}
num_train_examples = 10 # number of training examples

#generate some random data
X_train = standard_normal((num_train_examples, num_features))
y_train = randint(num_classes, size=num_train_examples)
X_test = standard_normal((num_train_examples, num_features))
y_test = randint(num_classes, size=num_train_examples)

#train classifiers - choose one
classifier = LinearSVC() #linear SVM (maximum margin perceptron)
classifier = MLPClassifier(alpha=0.1, max_iter=500)
classifier = KNeighborsClassifier(3)
classifier = DecisionTreeClassifier(max_depth=20) #single tee
classifier = RandomForestClassifier(n_estimators=30,max_depth=10) #30 trees
classifier = SVC(gamma=1, C=1) #SVM with RBF kernel

classifier.fit(X_train, y_train) #training stage
y_predicted = classifier.predict(X_test) #test stage
print('Accuracy = ', accuracy_score(y_test, y_predicted))

print(X_train)
print(y_train)
print(y_predicted)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = SVC(kernel='linear', C=3)  # C - softmargin
clf.fit(x_train, y_train)

print(f'Accuracy: {clf.score(x_test, y_test)}')

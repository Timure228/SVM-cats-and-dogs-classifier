import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

Categories = ['cats', 'dogs']
flat_data_arr = []  # input
target_arr = []  # output
datadir = r"C:\Users\Admin\PycharmProjects\pythonProject1\Data Science\Machine Learning\SVM"
for i in Categories:

    print(f'loading... category : {i}')  # put the images into num arrays
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))  # convert the picture into numbers
        img_resized = resize(img_array, (150, 150, 3))  # resize the picture
        flat_data_arr.append(img_resized.flatten())  # flatten() converts an array into 1 dimension
        target_arr.append(Categories.index(i))
    print(f'loaded Category: {i} successfully')

flat_data = np.array(flat_data_arr, dtype='object')
target = np.array(target_arr, dtype='object')

df = pd.DataFrame(flat_data)
df['Target'] = target

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

param_grid = {
    'C': [1, 10],
    'kernel': ['linear', 'poly']
}

svc = svm.SVC(probability=True)

model = GridSearchCV(svc, param_grid)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)

# Accuracy
print(f'Accuracy: {accuracy * 100}%')

print(classification_report(y_pred, y_test, target_names=['cat', 'dog']))


# Visualization
path = r'C:\Users\Admin\PycharmProjects\pythonProject1\Data Science\Machine Learning\SVM\cats\10.jpg'
img = imread(path)
plt.imshow(img)
plt.show()
img_resize = resize(img, [150, 150, 3])
l = [img_resize.flatten()]
probability = model.predict_proba(l)
for ind, val in enumerate(Categories):
    print(f'{val} = {int(probability[0][ind] * 100)}%')
print(f'The predicted image is: {Categories[model.predict(l)[0]]}')

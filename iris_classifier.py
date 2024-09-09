import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tables.tests.test_suite import test
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None)
data.columns = ['Sepal length', 'sepal width', 'Petal length', 'Petal width', 'Iris Type']
#讀取資料
#資料上沒有標頭，新增標頭

features = data.iloc[:, :4]
labels = data.iloc[:, 4:]
#分別讀取特徵與種類

oneHotEncoder = OneHotEncoder()
oneHotEncoder.fit(labels)
labels_Encoder = oneHotEncoder.transform(data[['Iris Type']]).toarray()
labels_Encoder
#把種類轉換成one hot vector

data[['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']] = labels_Encoder
data = data.drop(['Iris Type'], axis=1)
data
#表格增刪及編碼

x_train, x_test, y_train, y_test = train_test_split(features, labels_Encoder, test_size=0.2, random_state=50)
#將訓練資料做分割，並固定資料

model = tf.keras.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_dim=4),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
#建立模型
#兩層隱藏層的神經元皆設為40，活化函數設為ReLU
#輸出層活化函數設為softmax

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)
#採用Adam作為優化器
#損失函數使用Categorical Cross-entropy

train = model.fit(x_train, y_train, epochs=500, batch_size=10)
#訓練模型
#做500次Epoch，並且每次Epoch分別做12次Iteration

plt.plot(train.history['loss'])
plt.plot(train.history['accuracy'])
plt.legend(['Loss','Accuracy'])

plt.title('Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.show()
#繪製Loss及Accuracy

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_labels)
precisions = precision_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')

print("Accuracy=", accuracy)
print("Precisions=", precisions)
print("F1 Score=", f1)
#模型在測試資料上的表現
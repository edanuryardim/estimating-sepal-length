import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

from sklearn.linear_model import LinearRegression


df = pd.read_csv("IRIS.csv")
X=df["petal_length"]
y=df["sepal_length"]

X_train , X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y , train_size=0.8)
X_train = np.array(X_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_test = np.array(X_test)

X_train=X_train.reshape(-1,1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#model seçimi ve modelin eğitimi
lin_model = sklearn.linear_model.LinearRegression()
lin_model.fit(X_train, y_train)

inputfromconsole=2
X_predictions = lin_model.predict(X_train)
plt.plot(X_train,X_predictions)
plt.scatter(x=df["petal_length"],y=df["sepal_length"],color="pink")
plt.scatter(x=inputfromconsole,y=lin_model.predict([[inputfromconsole]]),s=100,color="red")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.title("Sepal length and petal length")
plt.show()


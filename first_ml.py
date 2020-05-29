import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#Load training dataset
url = "https://raw.githubusercontent.com/callxpert/datasets/master/data-scientist-salaries.cc"
names = ['Years-experience', 'Salary']
dataset = pandas.read_csv(url, names=names)
# shape
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
#visualize
dataset.plot()
plt.show()
X = dataset[['Years-experience']]
y = dataset['Salary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test,predictions))

print(model.predict(6.3))

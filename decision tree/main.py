from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


carseats = pd.read_csv('Carseats.csv')

carseats['Urban'] = carseats['Urban'].map({'Yes': 1, 'No': 0})
carseats['US'] = carseats['US'].map({'Yes': 1, 'No': 0})

x_train, x_test = train_test_split(carseats, test_size=0.2)

y_train = x_train['Urban']
x_train = x_train.drop('Urban',axis=1)
x_train = x_train.drop('ShelveLoc',axis=1)


y_test = x_test['Urban']
x_test = x_test.drop('Urban',axis=1)
x_test = x_test.drop('ShelveLoc',axis=1)

model = XGBClassifier()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


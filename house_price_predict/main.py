import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

housing = pd.read_csv('USA_Housing.csv')
housing = housing.drop('Address',axis=1)

price = housing['Price']
housing = housing.drop('Price',axis=1)
print(price)
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(1))


x_train, x_test, y_train, y_test = train_test_split(housing,
                                                    price,
                                                    test_size = 0.2,
                                                    random_state = 101)

model.compile(optimizer="Adam",loss="mse",metrics=['accuracy'])
model.fit(x_train,y_train,validation_data = (x_test, y_test.values),epochs=250,verbose=True)

prediction = model.predict(x_test)
print(prediction[:10],y_test.head())



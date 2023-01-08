# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %%
# importing data use .head() and .column.values to see headers
data = pd.read_csv("Life Expectancy Data.csv").dropna()

# %%
# check wheter the value matches and then convert it into boolean and finnally into interger
data["Status"] = data["Status"].isin(['Developed']).astype(int)
print(data.shape)

# %%
x = np.array(data.drop(['Country', 'Life expectancy '], axis=1))
y = np.array(data.iloc[:, 3])
print(x.shape)
print(y.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y)

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
score = regressor.score(X_test, y_test)
print(score)

# %%
y_predicted = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test,y_predicted)
print(rmse)



import numpy as np
from sklearn.linear_model import LinearRegression

x_train = np.array([
   
    [1400,3,4], #[size,bedroom,bathrooms]
    [1600,3,2],
    [1700,4,3], 
    [1935,4,2],
])

y_train = np.array([245000, 312000, 279000, 308000])    #[prices for house]

# To train call model Linear Regression

model = LinearRegression()

model.fit(x_train,y_train)

# Just trying new data to get predicted value

new_house = np.array([
    [1450,3,3]
])
predicted_price = model.predict(new_house)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
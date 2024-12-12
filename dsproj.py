import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
# Load the California Housing dataset
housing = fetch_california_housing()

# Create a DataFrame from the data
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual and predicted values
df.plot(figsize=(10, 6))
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Housing Prices')
plt.legend()
plt.show()

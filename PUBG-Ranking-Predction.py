import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_excel('pubg.xlsx')

# Select features and target variable
x = df[['Eliminations', 'Assists', 'Damage', 'Survived']]
y = df['Ranking']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Create a new DataFrame for prediction
new_df = pd.DataFrame({
    'Eliminations': [15],
    'Assists': [2],
    'Damage': [1600],
    'Survived': [12]
})

# Predict ranking for new data
prediction = model.predict(new_df)


prediction = np.maximum(prediction, 1)
print(f'Your Ranking will be: {int(prediction[0])}')

#

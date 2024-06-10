# CODTECHIT_INTERNSHIPTASK02_DAM
This repository contains the code and resources for my internship task 2.
# Report on Predictive Modeling with Linear Regression

## Project Name: Predictive Modeling with Linear Regression on the Palmer Penguins Dataset

### Conducted By: Janvi Deepak Bhanushali

### Platform: Jupyter Notebook

### Language Used: Python

### Libraries Used: 
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Organization: CodtechIT Solutions

### Internship Task: Task Two

---

## Task Overview

This task involves implementing a simple linear regression model using the Palmer Penguins dataset. The objective is to predict the `body_mass_g` of penguins based on their `flipper_length_mm`. The steps include splitting the data into training and testing sets, training the model on the training data, evaluating its performance using metrics like Mean Squared Error (MSE) and R-squared, and making predictions on the test set. Additionally, we will visualize the regression line and actual vs. predicted values to assess the model's accuracy.

## Implementation

### 1: Import Libraries and Load Dataset

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Palmer Penguins dataset
df = sns.load_dataset('penguins').dropna()
df.head()
```

The dataset is loaded using seaborn's `load_dataset` function, and missing values are dropped using `dropna()`. The first few rows of the dataset are displayed to understand its structure.

### 2: Select Features and Target Variable

```python
# Select the feature and target variable
X = df[['flipper_length_mm']]
y = df['body_mass_g']

# Display the first few rows
print(X.head())
print(y.head())
```

Here, we select `flipper_length_mm` as the feature and `body_mass_g` as the target variable. The first few rows of both the feature and target variables are displayed.

### 3: Split Data into Training and Testing Sets

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the splits
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Testing set: {X_test.shape}, {y_test.shape}')
```

The data is split into training and testing sets using an 80-20 split. The shapes of the training and testing sets are printed to verify the split.

### 4: Train the Linear Regression Model

```python
# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the model coefficients
print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
```

A Linear Regression model is created and trained on the training data. The model's coefficients (slope) and intercept are displayed.

### 5: Make Predictions on the Test Set

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the first few predictions
print(y_pred[:5])
```

The model makes predictions on the test set, and the first few predictions are displayed.

### 6: Evaluate Model Performance

```python
# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
```

The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics. These metrics are printed to assess the model's accuracy.

### 7: Visualize the Regression Line and Predictions

#### Regression Line on Training Data

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['flipper_length_mm'], y=y_train, label='Training data')
sns.lineplot(x=X_train['flipper_length_mm'], y=model.predict(X_train), color='red', label='Regression line')
plt.title('Regression Line on Training Data')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Body Mass (g)')
plt.legend()
plt.show()
```

A scatter plot of the training data is created, and the regression line is plotted to show the relationship between flipper length and body mass.

#### Actual vs Predicted Values on Test Data

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Body Mass (g)')
plt.ylabel('Predicted Body Mass (g)')
plt.show()
```

A scatter plot of the actual vs. predicted values on the test data is created. The diagonal line represents perfect predictions. Points closer to this line indicate more accurate predictions.

## Insights gained :

1. **Data Splitting**:
   - The dataset is split into training and testing sets with an 80-20 split, ensuring that the model is trained on a large portion of the data and evaluated on a separate portion.

2. **Model Training**:
   - The Linear Regression model is trained on the training data. The coefficient (slope) indicates how much the body mass increases with each mm increase in flipper length. The intercept represents the body mass when the flipper length is zero.

3. **Predictions**:
   - The model makes predictions on the test set based on the learned relationship between `flipper_length_mm` and `body_mass_g`.

4. **Model Evaluation**:
   - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values. A lower MSE indicates a better fit.
   - **R-squared (R²)**: Indicates the proportion of variance in the dependent variable (body mass) that is predictable from the independent variable (flipper length). An R² value closer to 1 indicates a better fit.

5. **Visualization**:
   - **Regression Line on Training Data**: The plot shows the training data points and the regression line, which represents the best-fit line according to the model.

**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK02_DAM/assets/171580805/de95044d-620b-4893-876e-a1fa6e4fae73)

   - **Actual vs. Predicted Values on Test Data**: The plot compares the actual body mass values with the predicted values. Points closer to the diagonal line indicate more accurate predictions.

**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK02_DAM/assets/171580805/18be61f5-98ac-48bd-957a-466a085ac0cd)

### Important  Outputs

- **Coefficient**: 49.67 (indicates how much body mass increases with each mm increase in flipper length)
- **Intercept**: -2972.32 (body mass when flipper length is zero, not realistic but part of the linear equation)
- **Mean Squared Error**: 19474.30 (average squared error between actual and predicted values)
- **R-squared**: 0.87 (87% of the variance in body mass is explained by flipper length)

By following these steps, we successfully implemented a simple linear regression model to predict the body mass of penguins based on their flipper length and evaluated its performance.

## Conclusion

The predictive modeling task using linear regression demonstrates how a simple statistical model can be used to understand the relationship between two continuous variables. The model's performance, as indicated by MSE and R-squared, shows a strong predictive capability, suggesting that flipper length is a good predictor of body mass in penguins. Visualization of the regression line and actual vs. predicted values helps in assessing the accuracy and reliability of the model.

This report covers the entire process of implementing, evaluating, and visualizing a simple linear regression model, providing insights into the relationship between flipper length and body mass in penguins.

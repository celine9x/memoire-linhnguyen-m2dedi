import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('encodage.csv')

# Display the first few rows of the data
print(data.head())

# Perform linear regression
# Define the model
model = smf.ols('intention_use ~ perception_AI_risk', data=data)

# Fit the model
results = model.fit()

# Print the summary of the regression
print(results.summary())

# Plotting the regression line
sns.lmplot(x='intention_use', y='perception_AI_risk', data=data, ci=None)
plt.xlabel('Intention to use AI')
plt.ylabel('Perception of AI risks')
plt.title('Linear Regression: Intention to use AI vs. Perception of AI risks')
plt.show()
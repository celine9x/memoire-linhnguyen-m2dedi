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
model = smf.ols('intention_use ~ AI_knowledge', data=data)

# Fit the model
results = model.fit()

# Print the summary of the regression
print(results.summary())

# Plotting the regression line
sns.lmplot(x='intention_use', y='AI_knowledge', data=data, ci=None)
plt.xlabel('Intention to use AI')
plt.ylabel('Knowledge about AI')
plt.title('Linear Regression: Knowledge about AI vs. Intention to use AI')
plt.show()
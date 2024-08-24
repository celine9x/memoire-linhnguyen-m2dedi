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
model = smf.ols('perception_AI_benefit ~ frequency_media_positive', data=data)

# Fit the model
results = model.fit()

# Print the summary of the regression
print(results.summary())

# Plotting the regression line
sns.lmplot(x='perception_AI_benefit', y='frequency_media_positive', data=data, ci=None)
plt.xlabel('Perception of AI benefits')
plt.ylabel('Positive framing about AI')
plt.title('Linear Regression: Perception of AI benefits vs. Positive framing about AI')
plt.show()
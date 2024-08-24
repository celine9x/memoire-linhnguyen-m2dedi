import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('encodage.csv')

# Print the first few rows of the dataframe to check the data
print(df.head())

# Prepare the data for the regression model
X = df[['traditional_media', 'digital_media']]
y = df['intention_use']

# Add a constant to the independent variables matrix (for the intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the regression
print(results.summary())

# Plotting the regression line
plt.figure(figsize=(12, 6))

# Plotting intention_use vs. traditional_media
plt.subplot(1, 2, 1)
sns.regplot(x='traditional_media', y='intention_use', data=df, ci=None)
plt.xlabel('Traditional Media')
plt.ylabel('Intention to Use AI')
plt.title('Regression: Intention to Use AI vs. Traditional Media')

# Plotting intention_use vs. digital_media
plt.subplot(1, 2, 2)
sns.regplot(x='digital_media', y='intention_use', data=df, ci=None)
plt.xlabel('Digital Media')
plt.ylabel('Intention to Use AI')
plt.title('Regression: Intention to Use AI vs. Digital Media')

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('regression_plots.png')

# Show the plots
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('cleaned_data.csv')  # Ensure dataset is cleaned

# Data Visualization
plt.figure(figsize=(8, 5))
sns.histplot(df['Value'], bins=30, kde=True, color='blue')
plt.title('Distribution of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig("histogram.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.boxplot(x=df['Category'], y=df['Value'], palette='Set2')
plt.title('Boxplot of Values by Category')
plt.xlabel('Category')
plt.ylabel('Value')
plt.savefig("boxplot.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig("heatmap.png")
plt.show()

print("Data Visualization Completed Successfully!")

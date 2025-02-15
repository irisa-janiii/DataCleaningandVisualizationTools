import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display uncleaned data
print("Uncleaned Data:")
print(df.head())

# -------------------- DATA CLEANING -------------------- #
df.drop(columns=['deck'], inplace=True)  # Drop 'deck' column due to too many missing values
df.dropna(subset=['age', 'embarked', 'fare'], inplace=True)  # Drop rows with missing values in 'age', 'embarked', 'fare'
df['embark_town'].fillna('Unknown', inplace=True)  # Fill missing values in 'embark_town' with 'Unknown'
df['age'].fillna(df['age'].median(), inplace=True)  # Fill missing age values with the median

# Adding a new column for age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Senior', 'Retired'])

# Adding a new column to indicate if the passenger was traveling alone
df['travel_alone'] = (df['sibsp'] + df['parch'] == 0).astype(int)

# *** Handling extreme values for 'fare' ***
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Limiting values outside the interquartile range
df['fare'] = np.where(df['fare'] < lower_bound, lower_bound, df['fare'])
df['fare'] = np.where(df['fare'] > upper_bound, upper_bound, df['fare'])

# *** Normalizing fare ***
df['fare_log'] = np.log1p(df['fare'])  # Natural log to reduce the impact of outliers

# *** Encoding categorical variables ***
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Male -> 0, Female -> 1
df['embark_town'] = df['embark_town'].astype('category').cat.codes  # Convert to numerical values

# General statistics
print("\nGeneral Statistics of the dataset:")
print(df.describe())

# Missing value analysis
print("\nNumber of missing values per column:")
print(df.isnull().sum())

# -------------------- VISUALIZATION -------------------- #

# Age distribution visualization
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Count plot of survivors by gender
plt.figure(figsize=(8,5))
sns.countplot(x='sex', hue='survived', data=df, palette='Set1')
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Boxplot of fare by passenger class
plt.figure(figsize=(8,5))
sns.boxplot(x='class', y='fare', data=df, palette='pastel')
plt.title("Fare by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Fare")
plt.show()

# Scatter plot of age vs fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='fare', hue='survived', data=df, palette='coolwarm', alpha=0.7)
plt.title("Age vs Fare on Titanic")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# Count plot of survivors by age group
plt.figure(figsize=(8,5))
sns.countplot(x='age_group', hue='survived', data=df, palette='Set2')
plt.title("Survival by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Count plot of survivors by travel type (alone vs with family)
plt.figure(figsize=(8,5))
sns.countplot(x='travel_alone', hue='survived', data=df, palette='coolwarm')
plt.xticks(ticks=[0, 1], labels=["With Family", "Alone"])
plt.title("Survival by Travel Type (Alone vs With Family)")
plt.xlabel("Travel Type")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Pairplot to see relationships between variables
sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue="survived", palette="husl")
plt.show()

# Heatmap to analyze correlation
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Variables")
plt.show()

# Pie chart of passenger gender distribution
plt.figure(figsize=(6,6))
df['sex'].replace({0: "Male", 1: "Female"}, inplace=True)  # Convert back to textual labels for better presentation
df['sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'pink'])
plt.title("Gender Distribution of Passengers")
plt.ylabel("")
plt.show()

# -------------------- SAVING DATA -------------------- #

# Save the cleaned data to a CSV file
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset has been saved to titanic_cleaned.csv")

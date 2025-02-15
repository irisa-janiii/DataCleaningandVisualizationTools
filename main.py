import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Shkarkimi i dataset-it Titanic nga seaborn
df = sns.load_dataset('titanic')

# Shfaqja e të dhënave të papastruara
print("Të dhënat e papastruara:")
print(df.head())

# Pastrimi i të dhënave
df.drop(columns=['deck'], inplace=True)  # Heqim kolonën 'deck' që ka shumë vlera të munguara
df.dropna(subset=['age', 'embarked'], inplace=True)  # Heqim rreshtat me vlera të munguara në 'age' dhe 'embarked'
df['embark_town'].fillna('Unknown', inplace=True)  # Mbushim vlerat e munguara me 'Unknown'
df['age'].fillna(df['age'].median(), inplace=True)  # Mbushim vlerat e munguara të moshës me medianën

# Statistika të përgjithshme
print("\nStatistika të përgjithshme të dataset-it:")
print(df.describe())

# Analizë e mungesave
print("\nNumri i vlerave të munguara në secilën kolonë:")
print(df.isnull().sum())

# Transformimi i të dhënave: Shtimi i një kolone të re për grupimin e moshave
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], labels=['Fëmijë', 'Adoleshent', 'I Rritur', 'I Moshuar', 'Pensionist'])

# Vizualizimi i shpërndarjes së moshës
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title("Shpërndarja e Moshës së Pasagjerëve të Titanic")
plt.xlabel("Mosha")
plt.ylabel("Frekuenca")
plt.show()

# Grafiku i numrit të mbijetuarve sipas gjinisë
plt.figure(figsize=(8,5))
sns.countplot(x='sex', hue='survived', data=df, palette='Set1')
plt.title("Mbijetesa sipas Gjinisë")
plt.xlabel("Gjinia")
plt.ylabel("Numri i Pasagjerëve")
plt.legend(title="Mbijetuar", labels=["Jo", "Po"])
plt.show()


# Boxplot i çmimit të biletës sipas klasës së udhëtimit
plt.figure(figsize=(8,5))
sns.boxplot(x='class', y='fare', data=df, palette='pastel')
plt.title("Çmimi i Biletës sipas Klasës së Pasagjerëve")
plt.xlabel("Klasa")
plt.ylabel("Çmimi i Biletës")
plt.show()

# Scatter plot i moshës dhe çmimit të biletës
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='fare', hue='survived', data=df, palette='coolwarm', alpha=0.7)
plt.title("Mosha dhe Çmimi i Biletës në Titanic")
plt.xlabel("Mosha")
plt.ylabel("Çmimi i Biletës")
plt.show()

# Përdorimi i një 'pairplot' për të parë lidhjet mes variablave
sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue="survived", palette="husl")
plt.show()

# Përdorimi i një 'heatmap' për të analizuar korrelacionin
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrica e Korrelacionit të Variablave")
plt.show()

# Pie chart për përqindjen e pasagjerëve sipas gjinisë
plt.figure(figsize=(6,6))
df['sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'pink'])
plt.title("Përqindja e Pasagjerëve sipas Gjinisë")
plt.ylabel("")
plt.show()

# Ruajtja e të dhënave të pastruara në një skedar CSV
df.to_csv("titanic_cleaned.csv", index=False)
print("\nDataset-i i pastruar është ruajtur në titanic_cleaned.csv")

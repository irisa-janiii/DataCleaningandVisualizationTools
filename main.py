import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Shkarkimi i dataset-it Titanic nga seaborn
df = sns.load_dataset('titanic')

# Shfaqja e të dhënave të papastruara
print("Të dhënat e papastruara:")
print(df.head())

# -------------------- PASRIMI I TË DHËNAVE -------------------- #
df.drop(columns=['deck'], inplace=True)  # Heqim kolonën 'deck' që ka shumë vlera të munguara
df.dropna(subset=['age', 'embarked', 'fare'], inplace=True)  # Heqim rreshtat me vlera të munguara në 'age', 'embarked', 'fare'
df['embark_town'].fillna('Unknown', inplace=True)  # Mbushim vlerat e munguara me 'Unknown'
df['age'].fillna(df['age'].median(), inplace=True)  # Mbushim vlerat e munguara të moshës me medianën

# Shtimi i një kolone të re për grupimin e moshave
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], labels=['Fëmijë', 'Adoleshent', 'I Rritur', 'I Moshuar', 'Pensionist'])

# Shtimi i një kolone për të treguar nëse pasagjeri udhëtonte vetëm
df['travel_alone'] = (df['sibsp'] + df['parch'] == 0).astype(int)

# *** Trajtimi i vlerave ekstreme për 'fare' ***
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Kufizimi i vlerave jashtë intervalit
df['fare'] = np.where(df['fare'] < lower_bound, lower_bound, df['fare'])
df['fare'] = np.where(df['fare'] > upper_bound, upper_bound, df['fare'])

# *** Normalizimi i çmimit të biletës ***
df['fare_log'] = np.log1p(df['fare'])  # Logaritëm natyror për të reduktuar ndikimin e outliers

# *** Kodimi i variablave kategorike ***
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Mashkull -> 0, Femër -> 1
df['embark_town'] = df['embark_town'].astype('category').cat.codes  # Konvertimi në numra

# Statistika të përgjithshme
print("\nStatistika të përgjithshme të dataset-it:")
print(df.describe())

# Analizë e mungesave
print("\nNumri i vlerave të munguara në secilën kolonë:")
print(df.isnull().sum())

# -------------------- VIZUALIZIMI -------------------- #

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

# Grafiku i mbijetuarve sipas grupmoshës
plt.figure(figsize=(8,5))
sns.countplot(x='age_group', hue='survived', data=df, palette='Set2')
plt.title("Mbijetesa sipas Grupmoshës")
plt.xlabel("Grupmosha")
plt.ylabel("Numri i Pasagjerëve")
plt.legend(title="Mbijetuar", labels=["Jo", "Po"])
plt.show()

# Grafiku i mbijetuarve sipas llojit të udhëtimit (vetëm vs me familje)
plt.figure(figsize=(8,5))
sns.countplot(x='travel_alone', hue='survived', data=df, palette='coolwarm')
plt.xticks(ticks=[0, 1], labels=["Me familje", "Vetëm"])
plt.title("Mbijetesa sipas Udhëtimit Vetëm vs Me Familje")
plt.xlabel("Lloji i Udhëtimit")
plt.ylabel("Numri i Pasagjerëve")
plt.legend(title="Mbijetuar", labels=["Jo", "Po"])
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
df['sex'].replace({0: "Male", 1: "Female"}, inplace=True)  # Rikthimi në etiketa tekstuale për paraqitje më të mirë
df['sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'pink'])
plt.title("Përqindja e Pasagjerëve sipas Gjinisë")
plt.ylabel("")
plt.show()

# -------------------- RUATJA E TË DHËNAVE -------------------- #

# Ruajtja e të dhënave të pastruara në një skedar CSV
df.to_csv("titanic_cleaned.csv", index=False)
print("\nDataset-i i pastruar është ruajtur në titanic_cleaned.csv")


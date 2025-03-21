## ***EXNO2DS***
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
   ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = '/content/drive/MyDrive/Colab Notebooks/titanic_dataset.csv' 
dt = pd.read_csv(file_path)
dt
```
![Screenshot 2025-03-21 223445](https://github.com/user-attachments/assets/5ca5db4d-98b0-480a-ab99-31b3ae6ba607)
```
dt.info()
```
![Screenshot 2025-03-21 223619](https://github.com/user-attachments/assets/983df788-2c7e-4d61-a161-c04b0c9e66d8)
```
num_rows, num_columns = dt.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
```
![Screenshot 2025-03-21 223658](https://github.com/user-attachments/assets/38d426ff-16c0-4964-a1dd-fe0e03ab5812)
```
dt.set_index('PassengerId', inplace=True)
dt.head()
```
![Screenshot 2025-03-21 223742](https://github.com/user-attachments/assets/a8601587-c6af-45e9-9203-207d35f86212)
```
dt.describe()
```

![Screenshot 2025-03-21 231303](https://github.com/user-attachments/assets/d886df60-e139-4a8a-9e43-1e619d12378e)

```
# Count the frequency of each unique value in the 'Survived' column
survived_counts = dt['Survived'].value_counts()
survived_percentages = dt['Survived'].value_counts(normalize=True) * 100
print("Survived Counts:")
print(survived_counts)
print("\nSurvived Percentages:")
print(survived_percentages)
```
![Screenshot 2025-03-21 223851](https://github.com/user-attachments/assets/0c222dee-5f52-4d56-8367-3cb037087ffb)

```
# Count the frequency of each unique value in the 'Sex' column
sex_counts = dt['Sex'].value_counts()
sex_percentages = dt['Sex'].value_counts(normalize=True) * 100
print("Sex Counts:")
print(sex_counts)
print("\nSex Percentages:")
print(sex_percentages)
```
![Screenshot 2025-03-21 223938](https://github.com/user-attachments/assets/9bd2c535-1e48-450a-9b68-ef8f3fd35e02)

```
# Count the frequency of each unique value in the 'Pclass' column
pclass_counts = dt['Pclass'].value_counts()
pclass_percentages = dt['Pclass'].value_counts(normalize=True) * 100
print("Pclass Counts:")
print(pclass_counts)
print("\nPclass Percentages:")
print(pclass_percentages)
```
![Screenshot 2025-03-21 224013](https://github.com/user-attachments/assets/cbff7a5b-872e-4947-ba05-57561b08696b)

```
# Count the frequency of each unique value in the 'Embarked' column
embarked_counts = dt['Embarked'].value_counts()
embarked_percentages = dt['Embarked'].value_counts(normalize=True) * 100
print("Embarked Counts:")
print(embarked_counts)
print("\nEmbarked Percentages:")
print(embarked_percentages)
```
![Screenshot 2025-03-21 230450](https://github.com/user-attachments/assets/4b4cebc0-2a46-45db-a6a4-3ae31e4db67a)

```
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for seaborn
sns.set(style="whitegrid")
# Plot the count of survivors
sns.countplot(x='Survived', data=dt)
plt.title('Survival Distribution')
plt.show()
```
![Screenshot 2025-03-21 224106](https://github.com/user-attachments/assets/2c56b9ee-4db5-4521-a2b9-1cb6974f93f6)

```
# Plot the count of passengers by class
sns.countplot(x='Pclass', data=dt)
plt.title('Passenger Class Distribution')
plt.show()
```
![Screenshot 2025-03-21 224201](https://github.com/user-attachments/assets/462e5ae5-77bb-46b8-966a-499917441643)

```
import seaborn as sns
import matplotlib.pyplot as plt
# Set the style for seaborn
sns.set(style="whitegrid")
# Create a count plot for the 'Survived' column
sns.countplot(x='Survived', data=dt)
plt.title('Survival Distribution (Univariate Analysis)')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
```
![Screenshot 2025-03-21 224238](https://github.com/user-attachments/assets/2656ec21-0e67-414f-80d6-b12f0e1e15e2)



```
# Identify unique values in the 'Pclass' column
unique_pclass = dt['Pclass'].unique()
print("Unique values in 'Pclass':", unique_pclass)
```
![Screenshot 2025-03-21 224327](https://github.com/user-attachments/assets/db735145-9172-4a34-a9d7-7097f464912f)

```
# Rename the 'Sex' column to 'Gender'
dt.rename(columns={'Sex': 'Gender'}, inplace=True)
# Verify the column name change
print(dt.columns)
```
![Screenshot 2025-03-21 224356](https://github.com/user-attachments/assets/75109efa-a35a-42c0-ab70-b9b69707d5f3)
```
import seaborn as sns
import matplotlib.pyplot as plt
# Use catplot to analyze the relationship between 'Survived' and 'Pclass'
sns.catplot(x='Pclass', hue='Survived', data=dt, kind='count', height=5, aspect=1.5)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Count')
plt.show()
```
![Screenshot 2025-03-21 224438](https://github.com/user-attachments/assets/5d3bb019-1d61-47d7-8321-23ec37211a01)

```
# Create a count plot for 'Pclass' with 'Survived' as the hue
fig, ax1 = plt.subplots(figsize=(8, 5))
graph = sns.countplot(x='Pclass', hue='Survived', data=dt, ax=ax1)
# Add labels to the bars
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2, height + 20.8, height, ha="center")
# Set plot title and labels
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right')
plt.show()
```
![Screenshot 2025-03-21 224512](https://github.com/user-attachments/assets/909f294f-b8fd-4c43-83bc-d1ba2e5d99d0)

```
# Create a boxplot for 'Age' vs 'Survived'
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=dt)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
```
![Screenshot 2025-03-21 224541](https://github.com/user-attachments/assets/e00bc6ca-20ce-4a83-8b90-13c4be622a72)

```
import seaborn as sns
import matplotlib.pyplot as plt
# Create a boxplot for 'Age' vs 'Pclass' with 'Gender' as the hue
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', hue='Gender', data=dt)
plt.title('Age Distribution by Passenger Class and Gender')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Age')
plt.legend(title='Gender')
plt.show()
```

![Screenshot 2025-03-21 224614](https://github.com/user-attachments/assets/88553610-4857-41c6-83ec-9687b86e00fc)
```
# Use catplot to analyze 'Pclass', 'Survived', and 'Gender'
sns.catplot(x='Pclass', hue='Survived', col='Gender', data=dt, kind='count', height=5, aspect=1)
plt.suptitle('Survival Count by Passenger Class and Gender', y=1.02)
plt.show()
```
![Screenshot 2025-03-21 224658](https://github.com/user-attachments/assets/656b79ba-3ceb-4106-a2f3-fb7736d894fe)
```
# Calculate the correlation matrix for numerical columns only
corr = dt.select_dtypes(include=np.number).corr()
# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```
![Screenshot 2025-03-21 231029](https://github.com/user-attachments/assets/ed0537a7-984c-4040-84b3-e7fb77d3b892)

```
# Create a pairplot
sns.pairplot(dt, hue='Survived', height=2.5)
plt.suptitle('Pairplot of Numerical Columns Colored by Survival Status', y=1.02)
plt.show()
```
![Screenshot 2025-03-21 231048](https://github.com/user-attachments/assets/bbb4054b-5dbb-4c7f-8e57-86082fa19647)




# RESULT
 Thus, the Exploratory Data Analysis on the given data set was performed successfully.

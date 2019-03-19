#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#get the data csv and combining them
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

#describing data
#columns of dataset
#print(train_df.columns.values)

#preview the data
#print(train_df.head())
#print(train_df.tail())

#print(train_df.info())

#print(train_df.describe())

#print(train_df.describe(include=['O']))

#pivot table using the class and if the passenger survived or not
#the results are grouped by the class and the mean is taken
#ordered by the survival rate
#print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_index(by='Survived', ascending=False))

#pivot table using the sex and if the passenger survived or not
#the results are grouped by the class and the mean is taken
#ordered by the survival rate
#print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_index(by='Survived', ascending=False))

#pivot table using the number of siblings/spouse and if the passenger survived or not
#the results are grouped by the class and the mean is taken
#ordered by the survival rate
#print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_index(by='Survived', ascending=False))

#pivot table, using the parch(family relations) and if the passenger survived or not
#the results are grouped by the class and the mean is taken
#ordered by the survival rate
#print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_index(by='Survived', ascending=False))


#chart showing how many people survived or not grouped by age
# g = sns.FacetGrid(train_df, col='Survived')
# print(g.map(plt.hist, 'Age', bins=20))
#plt.show()

#chart showing how many people survivied or not grouped by class
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#plt.show()

#chart showing the survival rate by class of each embarked gate
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete='deep')
# grid.add_legend()
# plt.show()

#chart showing the fare paid by each sex, split by gates and if survived or not
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()

#remove the columns ticket and cabin, because we determined that they are not  so important for analysis
#print('Before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

#print('After', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


#add the column title, take everything before the dot(.) as the title
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.')


#categorize the titles, transforming the selected titles as Rare
# for dataset in combine:
#   dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', \
#     'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Done'], 
#     'Rare')

  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(pd.crosstab(train_df['Title'], train_df['Sex']))

#the mean of survivability rate by titles
#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#categorize the titles, transforming the selected titles as Rare
# for dataset in combine:
#   dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', \
#     'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Done'], 
#     'Rare')

#the mean of survivability rate by titles
#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#categorize the titles, transforming the selected titles as Rare
for dataset in combine:
  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', \
    'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Done'], 
    'Rare')

#converting the titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#take the title, compare with the mapping, replace with the number, if not in mapping, put 0
for dataset in combine:
  dataset['Title'] = dataset['Title'].map(title_mapping)
  dataset['Title'] = dataset['Title'].fillna(0)

#print(train_df.head())

#the column name is not necessary anymore, because we have now the title normalized and ordered
#the column passengerid is not necessary too
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

#print(train_df.shape)
#print(test_df.shape)

#transform the column Sex from String to ordinal data
for dataset in combine:
  dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

#print(train_df.head())

#generate a random noise between the mean in fields with null or empty values, normalizing the data without interefering in the final result

#show the relation of age of passagens by class and sex
#grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()
#plt.show()

#generate an empty array for guessed Age by PclassXGender
guess_ages = np.zeros((2,3))
#print(guess_ages)

#iterate over sex(0,1) and Pclass(1,2,3) to calculate values of age for te six combinations
for dataset in combine:
  for i in range(0,2):
    for j in range(0,3):
      #drop the empty and null values of iterated sex and class
      guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

      #take the median value of ages of iterated sex and class
      age_guess = guess_df.median()

      #convert random age float to nearest .5 age
      guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

  
  for i in range(0,2):
    for j in range(0,3):
      #take the data where age is null, by sex and pclass and set guessed value
      dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

  #round to int
  dataset['Age'] = dataset['Age'].astype(int)

#print(train_df.head())

#create age bands
#five age bands
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#group the survive rates by age bands and show the mean values
#print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

#replace age with ordinals based on age band
for dataset in combine:
  dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
  dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
  dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
  dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
  dataset.loc[dataset['Age'] > 64, 'Age']

#print(train_df.head())

#remove the age band now that we have the ordinal age
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
#print(train_df.head())

#to allow the drop of columns Parch and SibSp we can create a column with the name FamilySize, suming both columns
for dataset in combine:
  dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#generates the mean of survivibility rate by the family size
#print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#create a column to know if the passenger is alone
for dataset in combine:
  dataset['IsAlone'] = 0
  dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#check the mean of survivibility rate if the person is alone or not
#print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#now is possible to drop the columns Parch, SibSp and FamilySize
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df]

#print(train_df.head())

#create an artificial column combining Pclass and age
for dataset in combine:
  dataset['Age*Class'] = dataset.Age * dataset.Pclass

#print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

#the column Embarked takes S, Q and C, based on gate of embarkation
#to fill the empty values in this column we can fill it with the most common ocurrence
#take the mode value of Embarked column
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
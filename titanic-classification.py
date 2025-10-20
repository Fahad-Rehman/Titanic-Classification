import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

#Preview train data
print(train.head())

#descriptive statistics
print(train.describe())

#check for missing values
print(train.isnull().sum().sort_values(ascending=False))


#separate target variable
Y = train['Survived']
train.drop(['Survived'], axis=1, inplace=True)

#concatenate train and test data for preprocessing
full_data = pd.concat([train, test], sort=False).reset_index(drop=True)

#handle missing values
#filling embarked with mode
full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace=True)

#filling fare with median based on Pclass
full_data['Fare'].fillna(full_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)

#dropping Cabin due to high missing values
full_data.drop(['Cabin'], axis=1, inplace=True)

#filling Age with median based on Pclass and Sex
#age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
full_data['Age'].fillna(full_data.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)

#verify no missing values remain
assert full_data.isnull().sum().sum() == 0, "There are still missing values!"


#Feature Engineering


#convert categorical features to numerical
full_data['Sex'] = full_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
full_data['Embarked'] = full_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

#new features
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = (full_data['FamilySize'] == 1).astype(int)

le = LabelEncoder()
full_data['Title'] = full_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
full_data['Title'] = le.fit_transform(full_data['Title'])

#Binning Age
full_data['AgeBin'] = pd.cut(full_data['Age'], bins=[0, 12, 20, 30, 40, 50, 60, 70, 80], labels=False)

#dropping unnecessary columns
full_data.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)

#split back into train and test sets
X = full_data.iloc[:len(Y), :]
X_test = full_data.iloc[len(Y):, :]
print("Processed training features shape:", X.shape)
print("Processed test features shape:", X_test.shape)



#split training data for validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, Y_pred)
print("Validation Accuracy:", accuracy)
#Final Prediction on test set
Y_test_pred = model.predict(X_test)

#Prepare submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': Y_test_pred.astype(int)
})
submission.to_csv('submission_updated.csv', index=False)

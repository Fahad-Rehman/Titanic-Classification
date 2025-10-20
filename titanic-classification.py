import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np



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

#Define models

SEED, FOLDS = 42, 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

def log_model():
    return LogisticRegression(max_iter=200, random_state=SEED)

def lgb_model():
    return LGBMClassifier(
        n_estimators=2000, learning_rate=0.01,
        num_leaves=8, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=SEED
    )

def xgb_model():
    return XGBClassifier(
        n_estimators=2000, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="binary:logistic",
        random_state=SEED, tree_method="hist", n_jobs=-1
    )

def cat_model():
    return CatBoostClassifier(
        iterations=2000, learning_rate=0.01, depth=4,
        l2_leaf_reg=3.0, random_seed=SEED,
        loss_function="Logloss", verbose=False
    )

def svm_model():
    return SVC(probability=True, kernel='rbf', C=1.0, random_state=SEED)

#Cross-validation and training

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

models = [("LOG", log_model()), ("LGBM", lgb_model()), ("XGB", xgb_model()), ("CAT", cat_model()), ("SVM", svm_model())]

oof_preds = {name: np.zeros(len(X)) for name, _ in models}
test_preds = {name: np.zeros(len(X_test)) for name, _ in models}

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, Y), 1):
    print(f"Fold {fold}...")
    X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_va = Y.iloc[train_idx], Y.iloc[valid_idx]
    X_tr_scaled, X_va_scaled = X_scaled[train_idx], X_scaled[valid_idx]

    for name, make_model in models:
        model = make_model
        if name in ["LOG", "SVM"]:
            model.fit(X_tr_scaled, y_tr)
            preds = model.predict_proba(X_va_scaled)[:, 1]
            test_preds[name] += model.predict_proba(X_test_scaled)[:, 1] / FOLDS
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            test_preds[name] += model.predict_proba(X_test)[:, 1] / FOLDS

        oof_preds[name][valid_idx] = preds

    print(f"  Fold {fold} done.")


#Blend predictions

stack_train = np.column_stack([oof_preds[name] for name, _ in models])
stack_test  = np.column_stack([test_preds[name] for name, _ in models])

meta = RidgeClassifier(alpha=1.0)
meta.fit(stack_train, Y)
final_preds = meta.predict(stack_test)



#Prepare submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_preds
})
submission.to_csv("submission_ensemble.csv", index=False)
print("Saved -> submission_ensemble.csv")


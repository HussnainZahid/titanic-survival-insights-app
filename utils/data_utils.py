import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def load_data(path='data/Titanic-Dataset.csv'):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.copy()
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    df.drop(columns=['Cabin'], inplace=True, errors='ignore')
    return df

def engineer_features(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

def apply_filters(df, filters):
    df_filtered = df[
        (df['Pclass'].isin(filters['Pclass'])) &
        (df['Sex'].isin(filters['Sex'])) &
        (df['Age'].between(filters['AgeMin'], filters['AgeMax'])) &
        (df['Fare'].between(filters['FareMin'], filters['FareMax'])) &
        (df['Embarked'].isin(filters['Embarked'])) &
        (df['Title'].isin(filters['Title']))
    ]
    return df_filtered

def perform_stats(df):
    results = {}
    if len(df) > 1:
        ctab = pd.crosstab(df['Sex'], df['Survived'])
        chi2, p, _, _ = stats.chi2_contingency(ctab)
        results['Chi-Square'] = f"Chi2(Sex vs Survival): {chi2:.2f}, p={p:.4f}"

        survived_age = df[df['Survived']==1]['Age']
        not_survived_age = df[df['Survived']==0]['Age']
        t_stat, p_val = stats.ttest_ind(survived_age, not_survived_age)
        results['T-Test'] = f"T-Test(Age): t={t_stat:.2f}, p={p_val:.4f}"

    return results

def detect_outliers(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

def train_ml_model(df):
    features = ['Pclass','Sex','Age','Fare','FamilySize']
    X = df[features].copy()
    y = df['Survived']
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, le, acc

def predict_survival(model, le, pclass, sex, age, fare, family_size):
    df = pd.DataFrame({
        'Pclass':[pclass],
        'Sex':[le.transform([sex])[0]],
        'Age':[age],
        'Fare':[fare],
        'FamilySize':[family_size]
    })
    return model.predict_proba(df)[0][1]

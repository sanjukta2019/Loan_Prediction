# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# data = pd.read_csv("train.csv")
# data.head()
#
# data.apply(lambda x: sum(x.isnull()),axis=0)
# data['Gender'].value_counts()
# data.Gender = data.Gender.fillna('Male')
# data['Married'].value_counts()
# data.Married = data.Married.fillna('Yes')
# data['Dependents'].value_counts()
# data.Dependents = data.Dependents.fillna('0')
# data['Self_Employed'].value_counts()
# data.Self_Employed = data.Self_Employed.fillna('No')
# data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
# data['Loan_Amount_Term'].value_counts()
# data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
# data['Credit_History'].value_counts()
# data.Credit_History = data.Credit_History.fillna(1.0)
# data.apply(lambda x: sum(x.isnull()),axis=0)
#
# # Splitting training data
# X = data.iloc[:, 1: 12].values
#
# #X = data.iloc[:, :10].values
# y = data.iloc[:, 12].values
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#
# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder
# labelencoder_X = LabelEncoder()
#
# for i in range(0, 5):
#     X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])
#
# X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])
#
# # Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y_train = labelencoder_y.fit_transform(y_train)
#
# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# for i in range(0, 5):
#     X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])
# X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])
# # Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y_test = labelencoder_y.fit_transform(y_test)
#
# # Applying PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)
# explained_variance = pca.explained_variance_ratio_
#
# # Fitting Logistic Regression to the Training set
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state = 0)
# clf.fit(X_train, y_train)
# # Predicting the Test set results
# y_pred = clf.predict(X_test)
#
# file = open('model.pkl', 'wb')
#
# import pickle
# pickle.dump(clf, file)
#
# # close the file
# file.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.head()

#checking for missing values
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#fill missing values
df_train['Gender'] = df_train['Gender'].fillna(
df_train['Gender'].dropna().mode().values[0] )
df_train['Married'] = df_train['Married'].fillna(
df_train['Married'].dropna().mode().values[0] )
df_train['Dependents'] = df_train['Dependents'].fillna(
df_train['Dependents'].dropna().mode().values[0] )
df_train['Self_Employed'] = df_train['Self_Employed'].fillna(
df_train['Self_Employed'].dropna().mode().values[0] )
df_train['LoanAmount'] = df_train['LoanAmount'].fillna(
df_train['LoanAmount'].dropna().median() )
df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna(
df_train['Loan_Amount_Term'].dropna().mode().values[0] )
df_train['Credit_History'] = df_train['Credit_History'].fillna(
df_train['Credit_History'].dropna().mode().values[0] )


#drop the uniques loan id
df_train.drop('Loan_ID', axis = 1, inplace = True)
df_train['Dependents'].value_counts()
df_train.info()

code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2, 'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}
df_train = df_train.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
df_test = df_test.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)

Dependents_ = pd.to_numeric(df_train.Dependents)
Dependents__ = pd.to_numeric(df_test.Dependents)

df_train.drop(['Dependents'], axis = 1, inplace = True)
df_test.drop(['Dependents'], axis = 1, inplace = True)

y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
y=pd.to_numeric(df_train.Loan_Status)

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
import pickle
file = open('model.pkl', 'wb')


pickle.dump(classifier, file)

# close the file
file.close()
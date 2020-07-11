import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

import numpy as np


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

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




code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2, 'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

df_train = df_train.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
#df_test = df_test.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)


y = df_train['Loan_Status']

X = df_train.drop('Loan_Status', axis = 1)
import numpy as np
def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train,test = data_split(df_train, 0.2)

X_train = train[['Gender','Married','Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']].to_numpy()
X_test = test[['Gender','Married','Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']].to_numpy()

Y_train = train[['Loan_Status']].to_numpy().reshape(492,)
Y_test = test[['Loan_Status']].to_numpy().reshape(122,)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, Y_train)
inputFeatures=[1,1,1,2,9083, 1447, 228, 360, 1, 2, 0]
infProb=model.predict_proba([inputFeatures])[0][1]
print(infProb)
import pickle
file = open('loan.pkl', 'wb')


pickle.dump(model, file)

# close the file
file.close()



# clf=LogisticRegression()
# clf.fit(X_train, Y_train)
# # inputFeatures=[104,1,22,1,1]
# # infProb=clf.predict_proba([inputFeatures])[0][1]
# #
# # print(infProb)
#
# file = open('model.pkl', 'wb')
#
#
# pickle.dump(clf, file)
#
# # close the file
# file.close()
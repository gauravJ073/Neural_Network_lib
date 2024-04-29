
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import read_csv, DataFrame, concat

dataframe = read_csv('Churn.csv')
array = dataframe.values
# separate array into input and output components
X = array[:,:-1]
Y = array[:,-1]
X_train,X_test, y_train, y_test=train_test_split(X, Y, stratify=Y, random_state=42, test_size=0.25);

# scaling dataset
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#convert back into dataframe
df_train=DataFrame(X_train)
Y_train=DataFrame(y_train)

# print(Y_train)
df_train=concat([df_train,Y_train], sort=False, axis=1)

df_test=DataFrame(X_test)
Y_test=DataFrame(y_test)

# print(Y_test)
df_test=concat([df_test,Y_test], sort=False, axis=1)

# print(df_test)

#saving to csv
df_train.to_csv('Churn_train.csv', index=False, header=False)
df_test.to_csv('Churn_test.csv', index=False, header=False)
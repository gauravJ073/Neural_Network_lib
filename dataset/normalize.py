
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame, concat

dataframe = read_csv('Iris.csv')
array = dataframe.values
# separate array into input and output components
X = array[:,0:-1]
Y = array[:,-1]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=5)
print(rescaledX[0:5,:])

#convert back into dataframe
df=DataFrame(rescaledX)
Y=DataFrame(Y)
print(Y)
df=concat([df,Y], sort=False, axis=1)

print(df)
#saving to csv
df.to_csv('norm_iris.csv', index=False, header=False)

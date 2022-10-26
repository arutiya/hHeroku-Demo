import pandas as pd
import numpy as np

df_new = pd.read_csv(r"C:\Users\91880\Documents\desktop folder\SD_new.csv")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df_new.columns

#df_new.drop(["Unnamed: 0"], axis = 1, inplace = True)


x = df_new.iloc[:,1:-1] #independent features
x = pd.get_dummies(x,drop_first = True)

y = df_new.iloc[:,-1] #Dependent features
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,test_size = 0.2)

from sklearn.ensemble import RandomForestRegressor

regressor1 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor1.fit(x, y)  

Y_pred = regressor1.predict(x_test)  # test the output by changing values
regressor1.score(x,y)

import pickle
pickle.dump(regressor1,open('model.pkl','wb'))






























#importing the necessary files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"App\Crop_recommendation.csv")

#lets split the dataset
Feature=df.iloc[:,0:-1]
Target=df.label
x_train,x_test,y_train,y_test=train_test_split(Feature,Target,test_size=0.3,random_state=42)

from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(x_train,y_train)

import pickle
pickle.dump(gb,open(r'App\model.pkl','wb'))
print(gb.predict([[50,60,39,4,43,54,78]]))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
data={
    'math':[70,45,65,60,40,32,20,55,48,60],
    'science':[20,45,67,49,39,57,69,55,24,50],
    'English':[30,47,48,58,67,87,64,56,58,60],
    'Result':['Fail','Fail','Fail','Fail','Fail','Fail','Fail','Pass','Fail','Pass']
}
df=pd.DataFrame(data)
df['Result']=df['Result'].map({'Pass':1,'Fail':0})
#train the model
x=df[['math','science','English']]
y=df['Result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(y_test)
#70% i/p 30% i/p,70% o/p,30% o/p
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')
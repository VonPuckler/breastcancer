import numpy as np
import pandas as pd


data = pd.read_csv("data.csv")
print("describe",data.describe())
print("info",data.info())
print(data)
data = data.drop(columns=["1000025"],axis=1)
data = data.rename(columns={"5":"s1","1":"s2","1.1":"s3","1.2":"s4","2":"s4","1.3":"s5","3":"s6","1.4":"s7","1.5":"s8","2.1":"s9"})
print(data)
data.replace("?",0,inplace=True)
print(data)
data['s1']=data['s1'].astype("float32")
data['s2']=data['s2'].astype("float32")
data['s3']=data['s3'].astype("float32")
data['s4']=data['s4'].astype("float32")
data['s5']=data['s5'].astype("float32")
data['s6']=data['s6'].astype("float32")
data['s7']=data['s7'].astype("float32")
data['s8']=data['s8'].astype("float32")
data['s9']=data['s9'].astype("float32")
print(data)

X=data.iloc[:,0:8].values
y=data.iloc[:,9].values

print(X)
print(y)
print("Shape of X",X.shape)
print("Shape of y",y.shape)

total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X[:train_length]
X_test=X[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.losses import BinaryCrossentropy
from keras.utils import np_utils

data.loc[data["s9"]=="2","s9"]=2
data.loc[data["s9"]=="4","s9"]=4

y_train=np_utils.to_categorical(y_train,num_classes=5)
x_train=np_utils.to_categorical(y_test,num_classes=5)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",x_train.shape)
print(y_train)
print(x_train)

Binary = BinaryCrossentropy()

model=Sequential()
model.add(Dense(500,input_dim=8,activation='softmax'))
model.add(Dense(300,activation='relu'))
model.add(Dense(100,activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(5,activation='relu'))
model.compile(loss=Binary,optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=5,epochs=15,verbose=1)
model.summary()

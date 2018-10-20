import numpy as np
import pandas as pd
import os
import time
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import ADASYN

import keras

from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.models import model_from_json

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

dataset = pd.read_csv("C:/Users/user/PythonProjects/document-data-extraction/data-extraction/training_testing/train3_wv_wsm.csv",header=None)
n=len(dataset.columns)-1

X = dataset.iloc[:,0:n]
Y = dataset.iloc[:,n].values
X, Y = ADASYN().fit_sample(X, Y)

Y1=[]
i=[0,0,0,0,0]
for y in Y:
    if(y=='AIMX'):
        Y1.append(0)
        i[0]=i[0]+1
    if(y=='BASE'):
        Y1.append(1)
        i[1]=i[1]+1
    if(y=='CONT'):
        Y1.append(2)
        i[2]=i[2]+1
    if(y=='MISC'):
        Y1.append(3)
        i[3]=i[3]+1
    if(y=='OWNX'):
        Y1.append(4)
        i[4]=i[4]+1
print(i)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y1, test_size = 0.30, random_state = 8)
print("Training size:", x_train.shape[0])
print("Testing size:", x_test.shape[0])

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train1=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test1=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

model=Sequential()
model.add(Conv1D(200, 5, padding='same',activation='relu',input_shape=(x_train.shape[1],1)))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(200, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(200, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()          

start_time = time.time()
model.fit(x_train1, y_train1, validation_data=(x_test1, y_test1),epochs=30, batch_size=150)
print("Total time to train:",(time.time() - start_time))

y_pred1 = model.predict(x_test1)
y_pred=np.argmax(y_pred1,axis=1)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(conf_mat)

accuracy=accuracy_score(y_pred,y_test)
print("Accuracy:",accuracy*100,'%')

model_json = model.to_json()
with open("train3_wv_wsm_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("train3_wv_wsm_model.h5")
print("Saved model to disk")
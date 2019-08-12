
# coding: utf-8

# In[1]:


# For compatibility
from __future__ import absolute_import
from __future__ import print_function
# For manipulating data
import pandas as pd
import numpy as np
from keras.utils import np_utils # For y values
import seaborn as sns
# For Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

data = pd.read_csv('raw.csv', sep=',',engine='python')
df = data

data_new = df.dropna(subset=['ASH','INSUL','TAAV','RH','dayav_ta','dayav_rh','MET'])
data_new.to_csv('raw_new.csv')
# get experimental data
lili_new = pd.read_csv('lili_feature.csv', sep=',',engine='python')
lili = lili_new[['clo','temperature','humidity','sensation']]
lili_y = lili['sensation']
lili_x = lili[['clo','temperature','humidity']]
# get features and thermal sensation
y = data_new['ASH']
#x = data_new[['INSUL','TAAV','RH','dayav_ta','dayav_rh','MET']]
x = data_new[['INSUL','TAAV','RH']]
y = np.round(y)
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
lili_encoded = lab_enc.fit_transform(lili_y)
y_train = np_utils.to_categorical(encoded,7)
xx = x.values.reshape(x.shape[0],x.shape[1],1)
lili_xx = lili_x.values.reshape(lili_x.shape[0],lili_x.shape[1],1)

nb_classes=7
#input_shape=x.iloc[0].shape
input_shape=xx[0].shape
model = Sequential()
model.add(LSTM(2048,return_sequences=True,input_shape=input_shape,dropout=0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(nb_classes, activation='sigmoid'))
#model.add(Dense(512,activation='relu',input_shape=input_shape))

optimizer = Adam(lr=1e-5,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
print(model.summary)

model.fit(xx,y_train)


y_test = model.predict(lili_xx)
pred = []
for i in range(len(lili_y)):
    pred.append(np.argmax(y_test[i]))

acc = 0
for i in range(len(lili_y)):
    if pred[i] == lili_encoded[i]:
        acc += 1


accuracy = acc/len(lili_encoded)

neutral_len_acc = len(lili_encoded[np.where(lili_encoded==3)])/len(lili_encoded)
print(neutral_len_acc)



# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:52:32 2021

@author: devan
"""

import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow import keras


#Loading saved model 
new_model = models.load_model('C:/Abhi/sem3/Neural Network/Group Project/code/NN_model')

new_model.summary()

#Loading CSV that has some testing data
df = pd.read_csv('C:/Abhi/sem3/Neural Network/Group Project/code/testing.csv')

df.drop('Unnamed: 0', axis=1, inplace= True)

#Poping label and saving it into labels variables
labels = df.pop('SUCCESSLEVEL')

#Transforming testing data(Input) into tensorflow
df = {key: value[:,tf.newaxis] for key, value in df.items()}
test_ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))


#Predction using model
y = new_model.predict(test_ds)



#----Showing the predictoing below----#
for i in range(len(y)):
    y[i] = y[i].argmax()

y_pred = y[:,:1]

y_pred = y_pred.astype('object')
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        y_pred[i] = 'un-successful'
    
    elif y_pred[i] == 1:
        y_pred[i] = 'Successful'
    
    else:
        y_pred[i] = 'In_progress'
        

y_act = labels.copy()
y_act = y_act.astype('object')
for i in range(len(y_act)):
    if y_act[i] == 0:
        y_act[i] = 'un-successful'
    
    elif y_act[i] == 1:
        y_act[i] = 'Successful'
    
    else:
        y_act[i] = 'In_progress'
        

for i in range(len(y_act)):
    print(f'Actual Value :{y_act[i]}  ---> Predict Value: {y_pred[i]}')
    print()















    


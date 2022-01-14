# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:37:32 2021

@author: Abhi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import seaborn as sns

df = pd.read_csv('C:/Abhi/sem3/Neural Network/Lab assignment 3/HYPE.csv')

df.info()


#Check type and total unique values for each column to get idea of
#Categorical and continuous column and to do feature selection
for i in list(df.columns):
    print(f'column name: {i}')
    ctype = df[i].dtype
    print(f'Type of Column: {ctype}')
    
    unique = df[i].nunique()
    print(f'Number of unique values in column: {unique}')
    
    print()
    
#By running above loop, it can be infered that some column has only one value
#in all records. This type of columns will not help in model training. 
#Hence droping this type of columns

df = df.drop(['RECORD COUNT','STUDENT TYPE NAME', 'STUDENT TYPE GROUP NAME','APPL EDUC INST TYPE NAME'], axis=1)

#Checking correlation between columns
#If columns have very high correlation that means, both columns will act same
#And, will help same to train the model. Hence, we can remove on of the column
#as duplicate.

df[['INTAKE TERM CODE','ADMIT TERM CODE']].corr()
df.drop('ADMIT TERM CODE', axis=1, inplace= True)

df[['PROGRAM SEMESTERS','TOTAL PROGRAM SEMESTERS']].corr()
df.drop('TOTAL PROGRAM SEMESTERS', axis=1, inplace= True)



df['APPL FIRST LANGUAGE DESC'].value_counts()

'''
APPL FIRST LANGUAGE DESC has below mentioned value_counts()
Unknown    406
English    250
Other        9

It can be seen that most of the values are unknown
And rest of the all values are nearly 'English'
Hence this column will not help to train the model and we can drop this column too
'''

df.drop('APPL FIRST LANGUAGE DESC', axis=1, inplace=True)



#---Converting label - 'SUCCESS LEVEL' into numeric values using (tensorflow)StringLookup---#
label = df['SUCCESS LEVEL']
vocab = ['Unsuccessful', 'Successful', 'In Progress']
data = tf.constant(label)
layer = tf.keras.layers.StringLookup()
layer.adapt(data)
label = layer(data)

l =[]
for i in list(label):
    l.append(i.numpy())
    
for i in range(len(l)):
    if l[i] ==1:
        l[i] = 0
    elif l[i] ==2:
        l[i] = 1
    else:
        l[i] = 2
    
labels = pd.Series(l)
df['SUCCESS LEVEL'] = labels



#Finding correlation between numeric columns and target columns
cor_col = df[['ID 2','PRIMARY PROGRAM CODE','EXPECTED GRAD TERM CODE','FIRST YEAR PERSISTENCE COUNT',
    'HS AVERAGE MARKS','ENGLISH TEST SCORE','SUCCESS LEVEL']].corr()

g=sns.heatmap(cor_col,annot=True,cmap="RdYlGn")

'''
from correlation matrix (cor_col), we can see that 2 columns have very low relation with
target column(SUCCESS LEVEL)

1) PRIMARY PROGRAM CODE
2) ENGLISH TEST SCORE
3) ID 2
    
Thus, both columns can be dropped
'''
df.drop(['ENGLISH TEST SCORE','PRIMARY PROGRAM CODE', 'ID 2'], axis=1, inplace=True)

#Last, HS AVERAGE GRADE and HS AVERAGE MARKS have high amount of missing values
#Henc, we can drop this columns

df.drop(['HS AVERAGE GRADE','HS AVERAGE MARKS'], axis=1, inplace= True)

df.drop(['FIRST GENERATION IND'], axis=1, inplace= True)


#Filling missing values


col_with_null = ['MAILING CITY NAME','MAILING POSTAL CODE GROUP 3',
                 'MAILING POSTAL CODE','MAILING PROVINCE NAME',
                 'MAILING COUNTRY NAME','AGE GROUP LONG NAME',
                 'APPLICANT CATEGORY NAME','PREV EDU CRED LEVEL NAME']

for i in col_with_null:
    df[i] = df[i].fillna(df[i].value_counts().index[0])
    
    

'''
  Converting String categorical columns to numbers using OrdinalEncoder
    Saving it to new dataframe - df2
    calculating correlation among features and target
    Removing features that has lower correlation     
'''     
from sklearn.preprocessing import OrdinalEncoder


enc = OrdinalEncoder()

df2 = enc.fit_transform(df)

df2 = pd.DataFrame(df2)

di = {}
for i,k in enumerate(df.columns):
    di[i] = k
    
df2.rename(columns= di, inplace= True)

g=sns.heatmap(df2.corr(),annot=True,cmap="RdYlGn")


cor_col = df2.corr()
cor = cor_col[['SUCCESS LEVEL']]


c = cor.loc[(cor['SUCCESS LEVEL'] > -0.05) & (cor['SUCCESS LEVEL'] < 0.05 )  ]


del_col = c.index

df.drop(list(del_col), axis= 1, inplace= True)



'''
Removing space in column names. to save the tensorflow model 
'''

li_col = list(df.columns)

for i in range(len(li_col)):
    no_space = li_col[i].replace(" ","")
    df.rename(columns={li_col[i]: no_space}, inplace= True)

    
    
    
train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])



def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('SUCCESSLEVEL')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

batch_size = 50
train_ds = df_to_dataset(train, batch_size=batch_size)


test_ds = df_to_dataset(test, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
      

                         
for keys,i in train_ds.take(1):
    
    for k,v in keys.items():
        print(k)
        print(v)
        print()
        

[(train_features, label_batch)] = train_ds.take(1)

print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['INTAKETERMCODE'])
print('A batch of targets:', label_batch )
    
    
    
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))
    
    
    
    
all_inputs = []
encoded_features = []

categorical_cols= ['INTAKECOLLEGEEXPERIENCE', 'SCHOOLCODE','PROGRAMLONGNAME',
                   'STUDENTLEVELNAME','TIMESTATUSNAME','RESIDENCYSTATUSNAME',
                   'FUNDINGSOURCENAME','MAILINGCITYNAME',
                   'MAILINGPROVINCENAME','GENDER',
                   'DISABILITYIND','MAILINGCOUNTRYNAME','CURRENTSTAYSTATUS',
                   'FUTURETERMENROL','ACADEMICPERFORMANCE','AGEGROUPLONGNAME',
                   'APPLICANTCATEGORYNAME','PREVEDUCREDLEVELNAME']


for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
    



int_categorical_col = ['EXPECTEDGRADTERMCODE',
                       'FIRSTYEARPERSISTENCECOUNT']
    

for header in int_categorical_col:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int32')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='int32',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)
  




def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer


conti_col = ['INTAKETERMCODE']

for header in conti_col:
    continuous_col = tf.keras.Input(shape=(1,), name=header, dtype='int32')
    num_layer = get_normalization_layer(header, train_ds)
                                              
    encoded_continuous_col = num_layer(continuous_col)
    all_inputs.append(continuous_col)
    encoded_features.append(encoded_continuous_col)


'''
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(21,   activation="elu")(all_features)
x = tf.keras.layers.Dense(3000, activation="elu")(x)
x = tf.keras.layers.Dense(1000, activation="elu")(x)
#x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)

'''
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dropout(0.3)(all_features)
x = tf.keras.layers.Dense(21,   activation="tanh")(x)
x = tf.keras.layers.Dense(3000, activation="tanh")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(1000, activation="elu")(x)
#
output = tf.keras.layers.Dense(3, activation='softmax')(x)
    
model = tf.keras.Model(all_inputs, output)

    
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.fit(train_ds, epochs=30, validation_data=val_ds)

model.evaluate(test_ds)

model.save('C:/Abhi/sem3/Neural Network/Group Project/code/NN_model')




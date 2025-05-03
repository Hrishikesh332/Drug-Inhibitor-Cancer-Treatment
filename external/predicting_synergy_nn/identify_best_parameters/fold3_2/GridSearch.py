# pip install --upgrade keras-hypetune
# https://github.com/cerlymarco/keras-hypetune

import time
start_time = time.time()

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from keras import models, layers
from kerashypetune import KerasGridSearch, KerasGridSearchCV, KerasRandomSearch, KerasRandomSearchCV
from keras.callbacks import EarlyStopping

df = pd.read_csv('fold3_train2.csv', header=0)
df = df.sample(frac=1) ## shuffle the rows
df.shape

x_train=df.iloc[:,0:33]
y_train=df['ZIP']

y_train=np.array(y_train).reshape(-1,1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(y_train)
out=scaler.transform(y_train)
y_train=pd.DataFrame(out)

df = pd.read_csv('fold3_valid2.csv', header=0)
df = df.sample(frac=1) ## shuffle the rows
df.shape

x_valid=df.iloc[:,0:33]
y_valid=df['ZIP']

y_valid=np.array(y_valid).reshape(-1,1)
out=scaler.transform(y_valid)
y_valid=pd.DataFrame(out)


def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den

def build_model(param):
    #create model
    model = models.Sequential()
    model.add(layers.Dense(units=param['unit1'], activation=param['activ1']))
    model.add(layers.Dropout(param['dpr1']))
    model.add(layers.Dense(units=param['unit2'], activation=param['activ2']))
    model.add(layers.Dropout(param['dpr2']))
    model.add(layers.Dense(units=param['unit3'], activation=param['activ3']))
    model.add(layers.Dropout(param['dpr3']))
    model.add(layers.Dense(1))
    
    model.compile(loss='mse', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1E-5), 
              metrics = ['mse', correlation])
        
    return model

param_grid = {
    'activ1': ['tanh', 'relu'], 
    'activ2': ['tanh', 'relu'],
    'activ3': ['tanh', 'relu'],
    'unit1': [512, 256, 128],    
    'unit2': [512, 256, 128],
    'unit3': [512, 256, 128],
    'dpr1': [0.5],
    'dpr2': [0.5],
    'dpr3': [0.5],
    'epochs': 50000, 
    'batch_size': 100
}

es = EarlyStopping(patience=10, verbose=1, min_delta=1E-5, monitor='val_correlation', mode='auto', restore_best_weights=True)
kht = KerasGridSearch(build_model, param_grid, monitor='val_correlation', greater_is_better=True)
kht.search(np.array(x_train), np.array(y_train), validation_data=(np.array(x_valid), np.array(y_valid)), callbacks=[es])

print('best param is')
print(kht.best_params)
print('best score is')
print(kht.best_score)

print("--- %s seconds ---" % (time.time() - start_time))


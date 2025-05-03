import time
start_time = time.time()

import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv('fold3_alltrain.csv', header=0)
df = df.sample(frac=1) ## shuffle the rows
df.shape

x=df.iloc[:,0:33]
y=df['ZIP']

y=np.array(y).reshape(-1,1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(y)
out=scaler.transform(y)
y=pd.DataFrame(out)

x_train = x
y_train = y

df = pd.read_csv('fold3_test.csv', header=0)
df = df.sample(frac=1) ## shuffle the rows
df.shape

x=df.iloc[:,0:33]
y=df['ZIP']

y=np.array(y).reshape(-1,1)
out=scaler.transform(y)
y=pd.DataFrame(out)

x_test = x
y_test = y

def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den

from keras import models, layers

def build_model(myx):
    
    #create model
    model = models.Sequential()
    model.add(layers.Dense(512, activation='tanh', input_shape=[myx.shape[1]]))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    
    # complie model
    model.compile(loss='mse', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1E-5), 
              metrics = ['mse', correlation])
    
    return model

model = build_model(x_train)
model.summary()

model_train=model.fit(x_train, y_train, epochs=50000, validation_data=(x_test, y_test), batch_size=100)

np.savetxt('train_correlation.csv', np.array(model_train.history['correlation']), delimiter=',')
np.savetxt('val_correlation.csv', np.array(model_train.history['val_correlation']), delimiter=',')
np.savetxt('train_mse.csv', np.array(model_train.history['mse']), delimiter=',')
np.savetxt('val_mse.csv', np.array(model_train.history['val_mse']), delimiter=',')

# import pickle
# pickle.dump(model, open('model_out.plk', 'wb'))
tf.keras.models.save_model(model, filepath='save.model3.50000')

print("--- %s seconds ---" % (time.time() - start_time))


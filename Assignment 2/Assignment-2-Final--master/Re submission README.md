# Assignment 2 
## Resubmission 

#define Model
from keras.layers import Activation
model = Sequential()

model.add(Convolution2D(10, 3, 3, activation='relu', input_shape=(28,28,1))) # 26
use_bias = False
model.add(BatchNormalization())

model.add(Convolution2D(16, 3, 3, activation='relu')) # 24
use_bias = False
model.add(BatchNormalization())
model.add(Convolution2D(20, 3, 3, activation='relu')) # 22
use_bias = False
model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2))) # 11
model.add(Convolution2D(10, 1, 1, activation='relu')) # 
use_bias = False
model.add(BatchNormalization())

model.add(Convolution2D(16, 3, 3, activation='relu')) # 9
use_bias = False
model.add(BatchNormalization())
model.add(Convolution2D(20, 3, 3, activation='relu')) # 7
use_bias = False
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(10, 1, activation='relu')) #7

model.add(Convolution2D(10, 7))
model.add(Flatten())
model.add(Activation('softmax'))
********************************************************************************************************************

model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_17 (Conv2D)           (None, 26, 26, 10)        100       
_________________________________________________________________
batch_normalization_13 (Batc (None, 26, 26, 10)        40        
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 24, 24, 16)        1456      
_________________________________________________________________
batch_normalization_14 (Batc (None, 24, 24, 16)        64        
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 22, 22, 20)        2900      
_________________________________________________________________
batch_normalization_15 (Batc (None, 22, 22, 20)        80        
_________________________________________________________________
dropout_5 (Dropout)          (None, 22, 22, 20)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 11, 11, 20)        0         
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 11, 11, 10)        210       
_________________________________________________________________
batch_normalization_16 (Batc (None, 11, 11, 10)        40        
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 9, 9, 16)          1456      
_________________________________________________________________
batch_normalization_17 (Batc (None, 9, 9, 16)          64        
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 7, 7, 20)          2900      
_________________________________________________________________
batch_normalization_18 (Batc (None, 7, 7, 20)          80        
_________________________________________________________________
dropout_6 (Dropout)          (None, 7, 7, 20)          0         
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 7, 7, 10)          210       
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 1, 1, 10)          4910      
_________________________________________________________________
flatten_3 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 14,510
Trainable params: 14,326
Non-trainable params: 184
_________________________________________________________________

# Epoch Logs

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 3s 47us/step - loss: 0.0209 - acc: 0.9934 - val_loss: 0.0471 - val_acc: 0.9885
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0134 - acc: 0.9954 - val_loss: 0.0296 - val_acc: 0.9927
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 3s 45us/step - loss: 0.0079 - acc: 0.9972 - val_loss: 0.0244 - val_acc: 0.9929
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 3s 43us/step - loss: 0.0057 - acc: 0.9981 - val_loss: 0.0262 - val_acc: 0.9934
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0043 - acc: 0.9987 - val_loss: 0.0265 - val_acc: 0.9935
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 3s 47us/step - loss: 0.0032 - acc: 0.9992 - val_loss: 0.0263 - val_acc: 0.9932
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0029 - acc: 0.9993 - val_loss: 0.0255 - val_acc: 0.9941
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 3s 47us/step - loss: 0.0022 - acc: 0.9993 - val_loss: 0.0233 - val_acc: 0.9941
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0023 - acc: 0.9995 - val_loss: 0.0289 - val_acc: 0.9929
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0024 - acc: 0.9992 - val_loss: 0.0240 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0023 - acc: 0.9994 - val_loss: 0.0285 - val_acc: 0.9942
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 3s 47us/step - loss: 0.0020 - acc: 0.9994 - val_loss: 0.0284 - val_acc: 0.9934
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 3s 45us/step - loss: 0.0020 - acc: 0.9995 - val_loss: 0.0289 - val_acc: 0.9938
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0019 - acc: 0.9995 - val_loss: 0.0268 - val_acc: 0.9941
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 3s 46us/step - loss: 0.0017 - acc: 0.9997 - val_loss: 0.0284 - val_acc: 0.9939
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 3s 45us/step - loss: 0.0019 - acc: 0.9995 - val_loss: 0.0273 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 3s 48us/step - loss: 0.0017 - acc: 0.9996 - val_loss: 0.0276 - val_acc: 0.9935
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 3s 48us/step - loss: 0.0017 - acc: 0.9996 - val_loss: 0.0277 - val_acc: 0.9940
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 3s 47us/step - loss: 0.0015 - acc: 0.9996 - val_loss: 0.0273 - val_acc: 0.9936
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 3s 45us/step - loss: 0.0017 - acc: 0.9996 - val_loss: 0.0290 - val_acc: 0.9935
<keras.callbacks.History at 0x7f2d25df49b0>

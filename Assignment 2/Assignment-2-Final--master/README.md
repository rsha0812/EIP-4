# Assignment-2-Final-
Program Code 9 DNN
from keras.layers import Activation
model = Sequential()

 
model.add(Convolution2D(10, 3, 3, activation='relu', input_shape=(28,28,1))) # 26
use_bias = False
model.add(BatchNormalization())

model.add(Convolution2D(16, 3, 3, activation='relu')) # 24
use_bias = False
model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, activation='relu')) # 22
use_bias = False
model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2))) # 11
model.add(Convolution2D(10, 1, 1, activation='relu')) # 11
use_bias = False
model.add(BatchNormalization())

model.add(Convolution2D(16, 3, 3, activation='relu')) # 9
use_bias = False
model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, activation='relu')) # 7
use_bias = False
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(10, 1, activation='relu')) #7
model.add(BatchNormalization())
model.add(Convolution2D(10, 7))
model.add(Flatten())
model.add(Activation('softmax'))
****************************************************************************************************************************

Total params: 13,278
Trainable params: 13,090
Non-trainable params: 188
************************************************************************************************************************
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 22s 361us/step - loss: 0.1739 - acc: 0.9452 - val_loss: 0.0898 - val_acc: 0.9699
Epoch 2/20
60000/60000 [==============================] - 20s 331us/step - loss: 0.0584 - acc: 0.9818 - val_loss: 0.0387 - val_acc: 0.9866
Epoch 3/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0462 - acc: 0.9853 - val_loss: 0.0423 - val_acc: 0.9861
Epoch 4/20
60000/60000 [==============================] - 20s 333us/step - loss: 0.0408 - acc: 0.9873 - val_loss: 0.0343 - val_acc: 0.9890
Epoch 5/20
60000/60000 [==============================] - 20s 336us/step - loss: 0.0372 - acc: 0.9883 - val_loss: 0.0333 - val_acc: 0.9887
Epoch 6/20
60000/60000 [==============================] - 20s 336us/step - loss: 0.0326 - acc: 0.9895 - val_loss: 0.0306 - val_acc: 0.9894
Epoch 7/20
60000/60000 [==============================] - 20s 334us/step - loss: 0.0309 - acc: 0.9898 - val_loss: 0.0407 - val_acc: 0.9866
Epoch 8/20
60000/60000 [==============================] - 20s 331us/step - loss: 0.0268 - acc: 0.9913 - val_loss: 0.0293 - val_acc: 0.9897
Epoch 9/20
60000/60000 [==============================] - 20s 335us/step - loss: 0.0265 - acc: 0.9912 - val_loss: 0.0283 - val_acc: 0.9911
Epoch 10/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0240 - acc: 0.9920 - val_loss: 0.0430 - val_acc: 0.9862
Epoch 11/20
60000/60000 [==============================] - 20s 330us/step - loss: 0.0239 - acc: 0.9920 - val_loss: 0.0371 - val_acc: 0.9886
Epoch 12/20
60000/60000 [==============================] - 20s 331us/step - loss: 0.0219 - acc: 0.9927 - val_loss: 0.0261 - val_acc: 0.9914
Epoch 13/20
60000/60000 [==============================] - 20s 336us/step - loss: 0.0219 - acc: 0.9927 - val_loss: 0.0279 - val_acc: 0.9907
Epoch 14/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0185 - acc: 0.9939 - val_loss: 0.0304 - val_acc: 0.9900
Epoch 15/20
60000/60000 [==============================] - 20s 333us/step - loss: 0.0189 - acc: 0.9939 - val_loss: 0.0278 - val_acc: 0.9907
Epoch 16/20
60000/60000 [==============================] - 20s 330us/step - loss: 0.0181 - acc: 0.9940 - val_loss: 0.0318 - val_acc: 0.9898
Epoch 17/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0166 - acc: 0.9946 - val_loss: 0.0401 - val_acc: 0.9885
Epoch 18/20
60000/60000 [==============================] - 20s 330us/step - loss: 0.0172 - acc: 0.9943 - val_loss: 0.0311 - val_acc: 0.9909
Epoch 19/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0166 - acc: 0.9947 - val_loss: 0.0386 - val_acc: 0.9889
Epoch 20/20
60000/60000 [==============================] - 20s 334us/step - loss: 0.0145 - acc: 0.9952 - val_loss: 0.0352 - val_acc: 0.9897
<keras.callbacks.History at 0x7f640fe1d4a8>
*********************************************************************************************************************************

score = model.evaluate(X_test, Y_test, verbose=0)
************************************************************************************************************

print(score)
[0.03518588825608931, 0.9897]
*************************************************************************************************************************

Acheived 99.4% accuracy from 16th epoch.

************************************************************************************************************
>> Decreased filter size to attain number of parameters under 15k
>> Tried different dropout value to avoid and overfitting and reduced the rate of validation loss 
>> Used batch normalisation to reduce loss gap between test and training data 
>> Able to get 99.4% Accuracy within 20 epochs by use using batch normalisation 
>> Avoiding biases helps in improving accuracy.

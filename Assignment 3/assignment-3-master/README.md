# assignment-3

# Define the model
model = Sequential()
model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3))) #nout(30) # RF(3)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(48, 3, 3)) #nout(28) # RF(5)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #nout(13) # RF(6)
model.add(Dropout(0.2))
model.add(SeparableConv2D(96, 3, 3, border_mode='same')) #nout(11) # RF(10)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(96, 3, 3)) #nout(9) # RF(14)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #nout(4) # RF(16)
model.add(Dropout(0.2))
model.add(SeparableConv2D(192, 3, 3, border_mode='same')) #nout(2) # RF(24)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(192, 3, 3)) #nout(0) # RF(32)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #nout(-1) # RF(36)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
****************************************************************************************************************************8

model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_13 (Separab (None, 32, 32, 48)        219       
_________________________________________________________________
activation_17 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_13 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
separable_conv2d_14 (Separab (None, 30, 30, 48)        2784      
_________________________________________________________________
activation_18 (Activation)   (None, 30, 30, 48)        0         
_________________________________________________________________
batch_normalization_14 (Batc (None, 30, 30, 48)        192       
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 15, 15, 48)        0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 15, 15, 48)        0         
_________________________________________________________________
separable_conv2d_15 (Separab (None, 15, 15, 96)        5136      
_________________________________________________________________
activation_19 (Activation)   (None, 15, 15, 96)        0         
_________________________________________________________________
batch_normalization_15 (Batc (None, 15, 15, 96)        384       
_________________________________________________________________
separable_conv2d_16 (Separab (None, 13, 13, 96)        10176     
_________________________________________________________________
activation_20 (Activation)   (None, 13, 13, 96)        0         
_________________________________________________________________
batch_normalization_16 (Batc (None, 13, 13, 96)        384       
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 6, 6, 96)          0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 6, 6, 96)          0         
_________________________________________________________________
separable_conv2d_17 (Separab (None, 6, 6, 192)         19488     
_________________________________________________________________
activation_21 (Activation)   (None, 6, 6, 192)         0         
_________________________________________________________________
batch_normalization_17 (Batc (None, 6, 6, 192)         768       
_________________________________________________________________
separable_conv2d_18 (Separab (None, 4, 4, 192)         38784     
_________________________________________________________________
activation_22 (Activation)   (None, 4, 4, 192)         0         
_________________________________________________________________
batch_normalization_18 (Batc (None, 4, 4, 192)         768       
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 2, 2, 192)         0         
_________________________________________________________________
dropout_13 (Dropout)         (None, 2, 2, 192)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 768)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 512)               393728    
_________________________________________________________________
activation_23 (Activation)   (None, 512)               0         
_________________________________________________________________
dropout_14 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 256)               131328    
_________________________________________________________________
activation_24 (Activation)   (None, 256)               0         
_________________________________________________________________
dropout_15 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_9 (Dense)              (None, 10)                2570      
=================================================================
Total params: 606,901
Trainable params: 605,557
Non-trainable params: 1,344
_________________________________________________________________

##Epoch Logs

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 31s 78ms/step - loss: 1.6232 - acc: 0.4141 - val_loss: 1.4704 - val_acc: 0.4956
Epoch 2/50
390/390 [==============================] - 27s 69ms/step - loss: 1.2163 - acc: 0.5674 - val_loss: 1.2597 - val_acc: 0.5636
Epoch 3/50
390/390 [==============================] - 27s 70ms/step - loss: 1.0383 - acc: 0.6350 - val_loss: 0.9039 - val_acc: 0.6846
Epoch 4/50
390/390 [==============================] - 27s 70ms/step - loss: 0.9317 - acc: 0.6737 - val_loss: 0.9185 - val_acc: 0.6779
Epoch 5/50
390/390 [==============================] - 27s 70ms/step - loss: 0.8479 - acc: 0.7058 - val_loss: 0.9036 - val_acc: 0.6830
Epoch 6/50
390/390 [==============================] - 27s 70ms/step - loss: 0.7945 - acc: 0.7244 - val_loss: 0.7769 - val_acc: 0.7312
Epoch 7/50
390/390 [==============================] - 27s 70ms/step - loss: 0.7405 - acc: 0.7432 - val_loss: 0.7454 - val_acc: 0.7436
Epoch 8/50
390/390 [==============================] - 27s 70ms/step - loss: 0.7002 - acc: 0.7562 - val_loss: 0.6831 - val_acc: 0.7656
Epoch 9/50
390/390 [==============================] - 27s 70ms/step - loss: 0.6625 - acc: 0.7690 - val_loss: 0.6805 - val_acc: 0.7683
Epoch 10/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6297 - acc: 0.7794 - val_loss: 0.8170 - val_acc: 0.7280
Epoch 11/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6024 - acc: 0.7903 - val_loss: 0.6943 - val_acc: 0.7648
Epoch 12/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5773 - acc: 0.7986 - val_loss: 0.8471 - val_acc: 0.7245
Epoch 13/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5501 - acc: 0.8078 - val_loss: 0.6403 - val_acc: 0.7850
Epoch 14/50
390/390 [==============================] - 27s 70ms/step - loss: 0.5323 - acc: 0.8119 - val_loss: 0.5897 - val_acc: 0.8025
Epoch 15/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5120 - acc: 0.8203 - val_loss: 0.6763 - val_acc: 0.7762
Epoch 16/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4940 - acc: 0.8262 - val_loss: 0.6562 - val_acc: 0.7879
Epoch 17/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4692 - acc: 0.8332 - val_loss: 0.6607 - val_acc: 0.7940
Epoch 18/50
390/390 [==============================] - 27s 70ms/step - loss: 0.4585 - acc: 0.8401 - val_loss: 0.6265 - val_acc: 0.7968
Epoch 19/50
390/390 [==============================] - 27s 70ms/step - loss: 0.4469 - acc: 0.8421 - val_loss: 0.6506 - val_acc: 0.7968
Epoch 20/50
390/390 [==============================] - 27s 70ms/step - loss: 0.4422 - acc: 0.8455 - val_loss: 0.6365 - val_acc: 0.7937
Epoch 21/50
390/390 [==============================] - 27s 70ms/step - loss: 0.4156 - acc: 0.8541 - val_loss: 0.6078 - val_acc: 0.8082
Epoch 22/50
390/390 [==============================] - 27s 70ms/step - loss: 0.4080 - acc: 0.8552 - val_loss: 0.6236 - val_acc: 0.8041
Epoch 23/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3998 - acc: 0.8580 - val_loss: 0.6380 - val_acc: 0.7988
Epoch 24/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3844 - acc: 0.8626 - val_loss: 0.5860 - val_acc: 0.8175
Epoch 25/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3748 - acc: 0.8690 - val_loss: 0.5718 - val_acc: 0.8211
Epoch 26/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3655 - acc: 0.8701 - val_loss: 0.6584 - val_acc: 0.7973
Epoch 27/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3610 - acc: 0.8715 - val_loss: 0.6193 - val_acc: 0.8097
Epoch 28/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3454 - acc: 0.8778 - val_loss: 0.6534 - val_acc: 0.8088
Epoch 29/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3474 - acc: 0.8784 - val_loss: 0.6098 - val_acc: 0.8095
Epoch 30/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3380 - acc: 0.8812 - val_loss: 0.6161 - val_acc: 0.8136
Epoch 31/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3235 - acc: 0.8856 - val_loss: 0.6290 - val_acc: 0.8121
Epoch 32/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3183 - acc: 0.8871 - val_loss: 0.6093 - val_acc: 0.8153
Epoch 33/50
390/390 [==============================] - 27s 70ms/step - loss: 0.3214 - acc: 0.8858 - val_loss: 0.6585 - val_acc: 0.8074
Epoch 34/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3084 - acc: 0.8897 - val_loss: 0.5972 - val_acc: 0.8240
Epoch 35/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3048 - acc: 0.8921 - val_loss: 0.6204 - val_acc: 0.8149
Epoch 36/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2937 - acc: 0.8959 - val_loss: 0.6065 - val_acc: 0.8168
Epoch 37/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2933 - acc: 0.8962 - val_loss: 0.6007 - val_acc: 0.8258
Epoch 38/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2853 - acc: 0.8996 - val_loss: 0.6147 - val_acc: 0.8232
Epoch 39/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2810 - acc: 0.9002 - val_loss: 0.6223 - val_acc: 0.8199
Epoch 40/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2793 - acc: 0.9021 - val_loss: 0.6671 - val_acc: 0.8137
Epoch 41/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2770 - acc: 0.9023 - val_loss: 0.6184 - val_acc: 0.8164
Epoch 42/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2684 - acc: 0.9043 - val_loss: 0.6242 - val_acc: 0.8218
Epoch 43/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2588 - acc: 0.9079 - val_loss: 0.6073 - val_acc: 0.8278
Epoch 44/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2627 - acc: 0.9086 - val_loss: 0.6145 - val_acc: 0.8233
Epoch 45/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2510 - acc: 0.9116 - val_loss: 0.6174 - val_acc: 0.8271
Epoch 46/50
390/390 [==============================] - 27s 70ms/step - loss: 0.2505 - acc: 0.9112 - val_loss: 0.6063 - val_acc: 0.8279
Epoch 47/50
390/390 [==============================] - 27s 69ms/step - loss: 0.2446 - acc: 0.9137 - val_loss: 0.6215 - val_acc: 0.8238
Epoch 48/50
390/390 [==============================] - 27s 69ms/step - loss: 0.2432 - acc: 0.9144 - val_loss: 0.6209 - val_acc: 0.8256
Epoch 49/50
390/390 [==============================] - 27s 69ms/step - loss: 0.2456 - acc: 0.9150 - val_loss: 0.6373 - val_acc: 0.8229
Epoch 50/50
390/390 [==============================] - 27s 69ms/step - loss: 0.2387 - acc: 0.9169 - val_loss: 0.6343 - val_acc: 0.8223
Model took 1362.57 seconds to train

Accuracy on test data is: 82.23

#!/usr/bin/env python
# coding: utf-8

# Installing required packages

# In[2]:


get_ipython().system(' pip install opencv.python')


# In[3]:


get_ipython().system(' pip install matplotlib')


# In[4]:


get_ipython().system(' pip install sklearn')


# Importing required libraries

# In[5]:


import tensorflow as tf
from tensorflow import keras


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# Loading our image datasets

# In[24]:


DATADIR ="C://Users//KAPIL VOHRA//Downloads//animals//animals"
CATEGORIES = ["dogs","cats","panda"]

for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array,cmap='gray')
        plt.show
        break
    break


# In[25]:


IMG_SIZE = 150

DATADIR ="C://Users//KAPIL VOHRA//Downloads//animals//animals"
CATEGORIES = ["dogs","cats","panda"]

training_data = []

for category in CATEGORIES:

    path = os.path.join(DATADIR, category)

    class_num = CATEGORIES.index(category)

    print(class_num)

    for img in os.listdir(path):

        try:

            img_array = cv2.imread(os.path.join(path,img))

            new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))

            training_data.append([new_array, class_num])

        except Exception as e:

            pass


# In[26]:


print(img_array)


# In[27]:


print(img_array.shape)


# In[28]:


print(len(training_data))


# In[29]:


import random

random.shuffle(training_data)


# In[30]:


X=[]
Y=[]


# In[31]:


for features, label in training_data:
    X.append(features) 
    Y.append(label)
    
X = np.array(X)

Y = np.array(Y)


# In[32]:


X.shape


# In[33]:


Y.shape


# Splitting dataset into train and test set

# In[35]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)


# Displaying some pictures to check the labels

# In[36]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i],cmap=plt.cm.binary)
    plt.xlabel(CATEGORIES[Y_train[i]])
plt.show()


# In[37]:


import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# Adding layers to the model 

# In[38]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))


# Compiling the model

# In[39]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


# Training the model

# In[40]:


history = model.fit(X_train, Y_train, epochs=15, validation_data=(X_test, Y_test))


# Evaluating model on test data

# In[41]:


model.evaluate(X_test,Y_test)


# In[42]:


model.summary()


# Predictions of images

# In[44]:


predictions=model.predict(X_test)


# In[45]:


predictions[0]


# In[46]:


np.argmax(predictions[0])


# In[47]:


Y_test[0]


# Testing the images

# In[48]:


def plot_image(i,predictions_array,true_label,img):
    predictions_array,true_label,img=predictions_array[i],true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img,cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label==true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel('{} {:2.0f}% ({})'.format(CATEGORIES[predicted_label],
                                         100*np.max(predictions_array),
                                         CATEGORIES[true_label]),
                                         color=color)


# In[49]:


i=1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,Y_test,X_test)


plt.show()


# In[53]:


i=49
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,Y_test,X_test)


plt.show()


# In[65]:


i=82
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,Y_test,X_test)


plt.show()


# In[66]:


num_rows=5
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range (num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions,Y_test,X_test)
plt.show()


# In[67]:


img=X_test[0]
print(img.shape)


# In[69]:


img=(np.expand_dims(img,0))
print(img.shape)


# In[70]:


predictions_single=model.predict(img)
print(predictions_single)


# In[71]:


np.argmax(predictions_single[0])


# Plotting training and validation accuracy and loss

# In[72]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()


# Confusion matrix

# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[74]:


Y_pred=model.predict(X_test)


# In[75]:


print(Y_pred)


# In[76]:


Y_pred=np.argmax(Y_pred,axis=1)


# In[77]:


print(Y_pred)


# In[78]:


cm=(confusion_matrix(Y_test,Y_pred))


# In[79]:


def plot_confusion_matrix(cm,classes,
                         normalize=False,
                         title='confusion matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('normalized confusion matrix')
    else:
        print('confusion matrix without normalization')
        
    print(cm)
    
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
        horizontalalignment="center",color="white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[80]:


cm_plot_labels=['dogs','cats','panda']
plot_confusion_matrix(cm,cm_plot_labels,title='confusion matrix')


# In[ ]:





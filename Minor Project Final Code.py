#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')


# In[3]:


paths = []
labels = []

for dirname, _, filenames in os.walk('C:\\Users\\devbe\\Downloads\\TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

print('Dataset is Loaded')  


# In[4]:


df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels


# In[5]:


counts = df['label'].value_counts()
ax = counts.plot.bar()
ax.set_xlabel('label')
ax.set_ylabel('count')
plt.show()


# In[6]:


def extract_mfcc(filename):
    y , sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc=40).T, axis=0)
    return mfcc


# In[7]:


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = [x for x in X_mfcc]
X = np.array(X)
X = np.expand_dims(X, -1)


# In[8]:


label_dict = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'ps':5, 'sad':6}
y = np.array([label_dict[label] for label in labels])
y = np.eye(len(label_dict))[y]


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.5),
    Dense(128, activation= 'relu'),
    Dropout(0.5),
    Dense(64, activation= 'relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[10]:


history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=512, shuffle=True)


# In[11]:


epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[12]:


loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[13]:


loss, accuracy = model.evaluate(X, y)
print(f'Test loss: {loss:.2f} - Test accuracy: {accuracy*100:.2f}%')


# In[15]:


# Split dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Get predictions for validation set
y_pred = model.predict(X_val)

# Convert one-hot encoded labels to integers
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[ ]:





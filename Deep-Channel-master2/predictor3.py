# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019
@author: ncelik34
"""
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import math


batch_size = 256

Qubname = 'outfinaltest987sined_SKM.csv'
Qub2name = 'outfinaltest987sined_halfamp.csv'
Dname = 'outfinaltest987sined.csv'

df30 = pd.read_csv(Dname, header=None)
dataset = df30.values
dataset = dataset.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
print(maxer)
maxeri = maxer.astype('int')
maxchannels = maxeri
idataset = dataset[:, 2]
idataset = idataset.astype(int)

scaler = MinMaxScaler(feature_range=(0, 1))
datasetUntransformed = dataset
dataset = scaler.fit_transform(dataset)


def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


train_size = int(len(dataset))

in_train = dataset[:, 1]
target_train = idataset
in_train = in_train.reshape(len(in_train), 1, 1, 1)

loaded_model = load_model('nmn_oversampled_deepchanel2_5.h5', custom_objects={
                          'mcor': mcor, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc})

loaded_model.summary()

# c = loaded_model.predict_classes(in_train, batch_size=batch_size, verbose=True)
c=np.argmax(loaded_model.predict(in_train, batch_size=batch_size, verbose =True),axis=1)

print(target_train[:20])
print(c[:20])

i_class1 = np.where(c == 1)[0]
i_class2 = np.where(target_train == 1)[0]
c=c.reshape(len(in_train),1)
#from sklearn.metrics import confusion_matrix
cm_dc = confusion_matrix(target_train, c)
#

df_qub = pd.read_csv(Qubname, header=None)
dataset_q = df_qub.values
idataset_q = dataset_q[:, 0]
idataset_q = idataset_q.astype(int)
cm_q = confusion_matrix(target_train, idataset_q)
#
#
df_qub2 = pd.read_csv(Qub2name, header=None)
dataset_q2 = df_qub2.values
idataset_q2 = dataset_q2[:, 0]
idataset_q2 = idataset_q2.astype(int)
cm_q2 = confusion_matrix(target_train, idataset_q2)


#cm24 = confusion_matrix(target_train, cn1)

# print(roc_auc_score(target_train,c))
print(Qubname)
print(Dname)
print("classification report of DC:")
print(classification_report(target_train, np.around(c)))
print("classification report of QuB SKM:")
print(classification_report(target_train, np.around(idataset_q)))
print("classification report of QuB half-amp:")
print(classification_report(target_train, np.around(idataset_q2)))
cm_dc = confusion_matrix(target_train, c)


lenny = 2000
ulenny = 3000
plt.figure(figsize=(30, 6))

plt.subplot(4, 1, 1)

plt.plot(dataset[lenny:ulenny, 1], color='blue', label="The Raw Data")
plt.title("The Raw Test")

plt.subplot(4, 1, 2)

plt.plot(target_train[lenny:ulenny], color='black', label="the actual idealisation")


plt.xlabel('timepoint')
plt.ylabel('current')
# plt.savefig(str(rnd)+'data.png')
#plt.savefig('destination_path.tiff', format='tiff', dpi=300)
plt.legend()
plt.show()

#New Attempt at better graphs -RRios

# # # Raw Data  - 5 channels 
plt.figure(figsize=(15,4))
plt.plot(datasetUntransformed[lenny:ulenny,1], color='black', label="The Raw Data")
plt.title("The Raw Test")
plt.xlabel('Time(ms)')
plt.ylabel('Current (pA)')
plt.legend(['Raw Current'], loc = 3)
plt.show()

# #Predicted Idealization 
plt.figure(figsize=(15, 4))
plt.plot(target_train[lenny:ulenny], color='blue', label="The Actual Idealisation")
plt.title("Gound Truth Idealization/Annotations")
plt.xlabel('Time(ms)')
plt.ylabel('No. of Open Channels')
plt.locator_params(axis="y", nbins = 5)
plt.legend(['Ground Truth'], loc = 3)
plt.show()

# #DEEPCHANNEL - Precticted Idealization 
plt.figure(figsize=(15,4))
plt.plot(c[lenny:ulenny], color='red', label="Deep-Channel")
plt.title("Deep-Channel Predicted Idealization")
plt.xlabel('Time(ms)')
plt.ylabel('No. of Open Channels')
plt.locator_params(axis="y", nbins = 5)
plt.legend(['Ground Truth'], loc = 3)
plt.show()

#Single Ion Channel Data 
# Raw Data  - 1 channel
plt.figure(figsize=(15, 4))
plt.plot(datasetUntransformed[lenny:ulenny,1], color='black', label="The Raw Data")
plt.title("The Raw Test")
plt.xlabel('Time(ms)')
plt.ylabel('Current (pA)')
plt.legend(['Raw Current'], loc = 2)
plt.show()

# #Predicted Idealization 
plt.figure(figsize=(15, 4))
plt.plot(target_train[lenny:ulenny], color='blue', label="The Actual Idealisation")
plt.title("Gound Truth Idealization/Annotations")
plt.xlabel('Time(ms)')
plt.ylabel('No. of Open Channels')
plt.locator_params(axis="y", nbins = 1)
plt.legend(['Ground Truth'], loc = 2)
plt.show()

# #DEEPCHANNEL - Precticted Idealization 
plt.figure(figsize=(15, 4))
plt.plot(c[lenny:ulenny], color='red', label="Deep-Channel")
plt.title("Deep-Channel Predicted Idealization")
plt.xlabel('Time(ms)')
plt.ylabel('No. of Open Channels')
plt.locator_params(axis="y", nbins = 1 )
plt.legend(['Ground Truth'], loc = 2)
plt.show()







# standard deviation of the dataset:
x_input = dataset[:, 1]
mean_x = sum(x_input) / np.count_nonzero(x_input)

sd_x = math.sqrt(sum((x_input - mean_x)**2) / np.count_nonzero(x_input))
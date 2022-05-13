import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

dataframe_list = list()


def data_chunks_creater(df_unsw_normal, df_bot_dos, df_bot_ddos):

    # we are taking 10 chunks only but you can take maximum upto 20
    number_of_chunks = 10
    data_limit = 2*80000*number_of_chunks

    # chunk_start and chunk_end are used to partition chunks from the oversampled data
    chunk_start = 0
    chunk_end = 2*80000

    # iterating so the chunks could be merged
    for _ in range(number_of_chunks):

        df_unsw_normal = df_unsw_normal

        df_bot_dos = pd.read_csv('dataset/train/bot_iot_dos.csv')
        df_bot_dos = df_bot_dos.iloc[:data_limit]
        df_bot_dos = df_bot_dos.iloc[chunk_start:chunk_end]

        df_bot_ddos = pd.read_csv('dataset/train/bot_iot_ddos.csv')
        df_bot_ddos = df_bot_ddos.iloc[:data_limit]
        df_bot_ddos = df_bot_ddos.iloc[chunk_start:chunk_end]

        # merging the data and also shuffling the data and appending the different chunks in a list, this also ensures the
        # the data is still into chunks its well balanced and consistent
        df = pd.concat([df_unsw_normal, df_bot_dos, df_bot_ddos],
                       ignore_index=True, sort=False)
        df = shuffle(df)
        df.reset_index(drop=True, inplace=True)
        dataframe_list.append(df)

        # updating chunk_start and chunk_end value
        chunk_start = chunk_end
        chunk_end = chunk_end+80000*2

    return dataframe_list


# A function which takes the big_data, but iterates through chunks, seperates x and y label and normalizes data using minmax
# the normalisation is done chunck wise and not for the full data, converting y_labels into 0,1,2
def data_preprocessing(data):
    x_label_data = list()
    y_label_data = list()
    for subset_data in data:
        df = subset_data
        x = df.loc[:, df.columns != 'category']
        # min_max
        x = (x-x.min())/(x.max()-x.min())
        df.category = pd.factorize(df.category)[0]
        y = df['category']
        x_label_data.append(x)
        y_label_data.append(y)
    return x_label_data, y_label_data


def data_preprocessing_test(data):
    x_label_data = list()
    y_label_data = list()
    df = data
    df = df.sample(frac=1, random_state=4)
    #df = df.sample(frac=1, random_state=4).reset_index(drop=True)
    x = df.loc[:, df.columns != 'category']
    # min_max
    x = (x-x.min())/(x.max()-x.min())
    df.category = pd.factorize(df.category)[0]
    y = df['category']
    x_label_data.append(x)
    y_label_data.append(y)
    return df, x_label_data, y_label_data

def data_preprocessing_test_something(data,random_state):
    x_label_data = list()
    y_label_data = list()
    df = data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    x = df.loc[:, df.columns != 'category']
    # min_max
    x = (x-x.min())/(x.max()-x.min())
    df.category = pd.factorize(df.category)[0]
    y = df['category']
    x_label_data.append(x)
    y_label_data.append(y)
    return df, x_label_data, y_label_data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

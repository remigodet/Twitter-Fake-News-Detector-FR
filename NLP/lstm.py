# imports
from collections import Counter
from nltk.corpus import stopwords
import re
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
from sklearn.datasets import load_files

from sklearn.model_selection import KFold
from keras import models
from keras import layers
import keras.backend
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow._api.v2 import data


def preprocess(X, y):
    '''
    :param X: tensor of values dtype:string encoding:utf-8 to preprocess
    Preprocessing: 
        - regular expression cleaning (including usernames for ethical reasons)
        - text formatting
        - lemmatization
        - sorting out empty values
    :param y: tensor of labels
    :outputs resX, resy: tensors tuple preprocessed
    '''
    stemmer = WordNetLemmatizer()
    resX = []
    resy = []
    for i in range(len(X)):
        text = X[i]
        label = y[i]
        text = text.decode('utf-8')
        # Remove all @username
        document = re.sub('^@.*? ', '', text)
        document = re.sub(' @.*? ', ' ', document)
        document = re.sub('@.*?$', '', document)
        # Remove all the special characters
        document = re.sub(r'\W', ' ', document)
        document = re.sub(r'http', ' ', document)
        document = re.sub(r'co', ' ', document)
        document = re.sub(r'pa', ' ', document)
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = [
            word for word in document if word not in stopwords.words('french')]
        document = ' '.join(document)
        if document != '':
            resX.append(document)
            resy.append(label)
    return resX, resy


def dataset_gen():
    '''
    :output X,y: dataset slices to generate the whole dataset with tensorflow tf.Datasets.from_generator
    Generates the dataset for the stored files in NLP/labeled_data/
    '''
    data = load_files("NLP/labeled_data")
    X, y = data.data, data.target
    X, y = preprocess(X, y)
    for i in range(len(y)):
        yield X[i], y[i]


def generate_Kfold_datasets(dataset, n_splits):
    '''
    :output train: next iter of kfold for training dataset
    :output test: next iter of kfold for testing dataset
    Generator of datasets in Kfold validation scheme.
    *Homebrew
    idea from SO to use sklearn as indices generator
    '''

    X = []
    Y = []
    for t in dataset:
        x, y = t
        X.append(x.numpy())
        Y.append(y.numpy())
    X = np.array(X)
    Y = np.array(Y)
    k_fold_indices = KFold(n_splits).split(X)

    for train_index, test_index in k_fold_indices:
        X_train = X[train_index]
        y_train = Y[train_index]
        X_test = X[test_index]
        y_test = Y[test_index]
        train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        yield train, test


def create_table(dataset):
    '''
    :param dataset: tensorflow dataset
    :output table: Tensorflow StaticVocabularyTable (CANNOT BE PICKLED ????, have to instanciate this as runtime for the app !)
    '''
    # voc

    vocab_size = 10000
    num_oov_buckets = 3000
    voc = Counter()

    for t in dataset:
        x, y = t
        voc.update(x.numpy().decode('utf-8').split())

    voc = [word for word, count in voc.most_common()[:vocab_size]]

    words = tf.constant(voc)
    word_ids = tf.range(len(voc), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)

    return tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


def train_model(dataset):
    '''
    :param dataset: tensorflow dataset

    Training the model based on kfold training to get a better approximation of the accuracy with fewer data.
    '''
    table = create_table(dataset)

    def encoder(x, y):
        '''
        returns the tensor with the text string embedded as numerical vector
        :outputs model, table: debugging reasons
        '''
        return table.lookup(x), y
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1

    for train, test in generate_Kfold_datasets(dataset, num_folds):

        train = train.batch(batch_size)
        train = train.map(encoder)
        test = test.batch(batch_size)
        test = test.map(encoder)
        vocab_size = 10000
        num_oov_buckets = 3000
        # model
        K = keras.backend
        inputs = layers.Input(shape=[None])
        embed = layers.Embedding(
            vocab_size+num_oov_buckets, embed_size)(inputs)
        # mask = layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
        l = layers.GRU(512, return_sequences=True)(embed)
        l = layers.Attention(use_scale=True)([l, l])
        l = layers.GRU(512)(l)
        outputs = layers.Dense(1, activation='sigmoid')(l)
        model = models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(loss=loss_function,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(train,
                            batch_size=batch_size,
                            epochs=nb_epochs,
                            verbose=verbosity)

        scores = model.evaluate(test, verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        # confusion matrix
        y_true = []
        y_pred = []
        if True:
            for item in test:
                x, y = item
                y = y.numpy()
                x = model.predict(x)
                y.flatten()
                x.flatten()
                for i in range(len(y)):
                    if x[i] > 0.68:
                        y_true.append(y[i])
                        y_pred.append(1)
                    elif x[i] < 0.32:
                        y_true.append(y[i])
                        y_pred.append(0)
            con_mat = confusion_matrix(y_true, y_pred)
            figure = plt.figure(figsize=(8, 8))
            sns.heatmap(con_mat, annot=True, cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        # Increase fold number
        fold_no = fold_no + 1
    return model, table


class LSTM_model:
    '''
    Homebrew model class to pack model and VocabularyTable altogether.
    def predict (self, text) is used to predict the credibility of a single piece of text
    Can be accessed with get_model from the other modules

    '''

    def __init__(self, model, table):
        self.model = model
        self.voc = table

    def predict(self, text):
        def process(text):
            # Remove all @username
            document = re.sub('^@.*? ', '', text)
            document = re.sub(' @.*? ', ' ', document)
            document = re.sub('@.*?$', '', document)
            # Remove all the special characters
            document = re.sub(r'\W', ' ', document)
            document = re.sub(r'http', ' ', document)
            document = re.sub(r'co', ' ', document)
            document = re.sub(r'pa', ' ', document)
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)

            # Converting to Lowercase
            document = document.lower()

            # Lemmatization
            document = document.split()
            stemmer = WordNetLemmatizer()
            document = [stemmer.lemmatize(word) for word in document]
            document = [
                word for word in document if word not in stopwords.words('french')]
            document = ' '.join(document)
            document = tf.constant([document])
            document = self.voc.lookup(document)
            return document

        text = process(text)
        return self.model.predict(text)[0][0]


def get_model():
    '''
    :output model: class LSTM_model for making predictions at runtime.
    '''
    dataset = tf.data.Dataset.from_generator(
        dataset_gen, (tf.string, tf.int32))
    model = models.load_model('NLP/models/lstm_model')
    table = create_table(dataset)
    lstm = LSTM_model(model, table)
    return lstm


if __name__ == '__main__':
    # to train the model !
    nltk_download('wordnet')
    nltk_download('stopwords')
    # hyperparameters NOT OPTIMAL :/
    num_folds = 2
    batch_size = 5
    nb_epochs = 5
    verbosity = 0
    gru_units = 32
    embed_size = 128
    loss_function = 'binary_crossentropy'
    optimizer = 'adam'

    # dataset instanciated
    dataset = tf.data.Dataset.from_generator(
        dataset_gen, (tf.string, tf.int32))
    # train
    model, table = train_model(dataset)
    model.summary()

    # # saving and loading
    model.save('NLP/models/lstm_model')  # reusable
    # model2 = models.load_model('NLP/models/lstm_model')
    # table = create_table(dataset)
    # lstm = LSTM_model(model2, table)
    # lstm = get_model()
    # print(lstm.predict('test text'))

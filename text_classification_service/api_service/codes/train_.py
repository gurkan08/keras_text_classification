
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os

from api_service.codes.params import Params
from api_service.codes.preprocess import *
from api_service.codes.model import ClassificationModel

class Main(object):

    sentence_tokenizer = None
    label_tokenizer = None

    def __init__(self):
        pass

    @staticmethod
    def create_data(data_dir):
        texts = []
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                text = ""
                with open(os.path.join(root, name), encoding="utf-8") as fp:
                    lines = fp.readlines()
                for line in lines:
                    text += line.rstrip().lower()
                text = " ".join(text.split())
                label = os.path.basename(root).rstrip().lower()
                texts.append((text, label))
        data = pd.DataFrame(texts, columns=["text", "label"])
        return data

    @staticmethod
    def read_dataset(data_dir):
        data = pd.read_excel(data_dir)
        return data

    @staticmethod
    def find_max_sentence_size(data):
        data["text"] = data["text"].map(lambda x: len(str(x).split()))
        Params.max_sent_size = int(data["text"].mean(axis=0)) # used mean sentence len

    @staticmethod
    def fasttext_embedding_init():
        # keras embeddings_initializer = 'uniform'
        embedding_matrix = np.zeros((len(Main.sentence_tokenizer.word_index) + 1, Params.embed_size)) # +1:zero_pad
        embedding_matrix[0,:] = np.random.uniform(size=(Params.embed_size,)) # 0 index for padding_value
        # for other vocab words, (UNK 1), 1. index == UNK word !
        for key, value in Main.sentence_tokenizer.word_index.items():
            embedding_matrix[value,:] = Params.fasttext_model.get_word_vector(key)
        Params.embedding_matrix = embedding_matrix # update embedding_matrix

    @staticmethod
    def save_plot(history):
        # loss figure
        plt.clf()
        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='valid')
        plt.title('train-valid loss')
        plt.ylabel('categorical_crossentropy loss')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "loss.png"))

        # accuracy figure
        plt.clf()
        plt.plot(history['acc'], label='train')
        plt.plot(history['val_acc'], label='valid')
        plt.title('train-valid accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(Params.plot_dir, "accuracy.png"))

    @staticmethod
    def run_preprocess(data):
        # preprocess
        data = lowercase(data)
        data = remove_punctuations(data)
        data = remove_numbers(data)
        data = remove_stop_words(data)
        #data = zemberek_stemming(data)
        data = first_5_char_stemming(data)
        data = data_shuffle(data, Params.shuffle_count)

        # split train-test
        X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"],
                                                            test_size=Params.test_size,
                                                            random_state=42,
                                                            stratify=data["label"])

        # max sentence size
        Main.find_max_sentence_size(pd.DataFrame(X_train, columns=["text"]))
        #print("mean sentence size --> ", Params.max_sent_size)

        # train data
        train_df = pd.DataFrame(zip(X_train, y_train), columns=["text", "label"])
        Main.sentence_tokenizer = Tokenizer(oov_token="UNK",
                                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                            lower=True)  # 0 index reserved as padding_value
        Main.sentence_tokenizer.fit_on_texts(train_df["text"])
        train_sentences = Main.sentence_tokenizer.texts_to_sequences(train_df["text"]) # list
        train_sentences = pad_sequences(train_sentences, maxlen=Params.max_sent_size, padding="post", value=0.)
        with open(os.path.join(Params.model_dir, "sentence_tokenizer.pickle"), "wb") as handle:
            pickle.dump(Main.sentence_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        Main.label_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                         lower=True)
        Main.label_tokenizer.fit_on_texts(train_df["label"])
        train_labels = Main.label_tokenizer.texts_to_sequences(train_df["label"]) # list
        train_labels = np.array(train_labels)
        train_labels = [to_categorical(i - 1, num_classes=len(Main.label_tokenizer.word_index)) for i in train_labels]
        train_labels = np.array(train_labels)
        train_labels = train_labels.reshape((train_labels.shape[0], train_labels.shape[-1])) # [n_samples, n_labels]
        with open(os.path.join(Params.model_dir, "label_tokenizer.pickle"), "wb") as handle:
            pickle.dump(Main.label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # test data
        test_df = pd.DataFrame(zip(X_test, y_test), columns=["text", "label"])
        test_sentences = Main.sentence_tokenizer.texts_to_sequences(test_df["text"]) # list
        test_sentences = pad_sequences(test_sentences, maxlen=Params.max_sent_size, padding="post", value=0.)

        test_labels = Main.label_tokenizer.texts_to_sequences(test_df["label"])
        test_labels = np.array(test_labels)
        test_labels = [to_categorical(i - 1, num_classes=len(Main.label_tokenizer.word_index)) for i in test_labels] # list
        test_labels = np.array(test_labels)
        test_labels = test_labels.reshape((test_labels.shape[0], test_labels.shape[-1])) # [n_samples, n_labels]

        # fasttext embedding init
        Main.fasttext_embedding_init()

        return train_sentences, train_labels, test_sentences, test_labels

    @staticmethod
    def run_train(X_train, y_train):
        model_obj = ClassificationModel(max_sentence_size=Params.max_sent_size,
                                        embed_size=Params.embed_size,
                                        vocab_size=len(Main.sentence_tokenizer.word_index) + 1,
                                        lstm_units=Params.lstm_units,
                                        dense_size=Params.dense_size,
                                        label_size=len(Main.label_tokenizer.word_index))
        model = model_obj.get_model()
        adam = Adam(lr=Params.lr, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
        print("------------model summary-------------")
        print(model.summary())

        # split train-valid
        # validation_split=Params.validation_split # dataset sonundan % x'i valid olarak alıyor, yanlış yöntem !
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=Params.validation_split,
                                                              random_state=42,
                                                              stratify=y_train)
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=Params.early_stop_patience)
        ]
        history = model.fit(np.array(X_train),
                            np.array(y_train),
                            batch_size=Params.batch_size,
                            epochs=Params.epochs,
                            validation_data=(X_valid, y_valid),
                            verbose=1,
                            shuffle=True,
                            callbacks=my_callbacks)
        model.save(os.path.join(Params.model_dir, "model.h5"))
        print("-------history---------")
        print(history.history)

        Main.save_plot(history.history)

        return model

    @staticmethod
    def run_test(model, X_test, y_test):
        pred_prob = model.predict(X_test) # [n_sample, n_class]

        def lambda_one_hot(pred_prob): #1-D np array
            class_index = np.argmax(pred_prob)
            pred_prob[:] = 0
            pred_prob[class_index] = 1
            return pred_prob
        pred_one_hot = np.array(list(map(lambda_one_hot, pred_prob)))

        accuracy = accuracy_score(y_test, pred_one_hot)
        f1_macro = f1_score(y_test, pred_one_hot, average="macro")
        f1_micro = f1_score(y_test, pred_one_hot, average="micro")
        f1_weighted = f1_score(y_test, pred_one_hot, average="weighted")
        print("test accuracy: ", accuracy)
        print("test f1 macro: ", f1_macro)
        print("test f1 micro: ", f1_micro)
        print("test f1 weighted: ", f1_weighted)

if __name__ == '__main__':

    """
    # create data.xlsx code
    data_dir = "data/news"
    labels = os.listdir(data_dir)
    #print(class_names)
    label2id = {label: id for id, label in enumerate(labels)}
    #print(label2id)
    data = Main.create_data(data_dir) # df
    #print(data)
    data.to_excel("data_6_class_and_other_balanced.xlsx", encoding="utf-8")
    """

    data_dir = "data_6_class_and_other_balanced.xlsx"
    data = Main.read_dataset(data_dir)
    X_train, y_train, X_test, y_test = Main.run_preprocess(data)
    model = Main.run_train(X_train, y_train)
    Main.run_test(model, X_test, y_test)
    
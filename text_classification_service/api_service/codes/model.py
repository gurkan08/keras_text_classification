
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model

from api_service.codes.params import Params

class ClassificationModel(object):

    def __init__(self,
                 max_sentence_size,
                 embed_size,
                 vocab_size,
                 lstm_units,
                 dense_size,
                 label_size):
        # mask_zero=True zero_padding, trainable=False fasttext embedding init
        input_layer = Input(shape=(max_sentence_size,))
        embed_layer = Embedding(input_dim=vocab_size,
                                output_dim=embed_size,
                                mask_zero=True,
                                weights=[Params.embedding_matrix],
                                trainable=False)(input_layer)
        lstm_layer = LSTM(lstm_units,
                          dropout=Params.drop_out,
                          recurrent_dropout=Params.drop_out,
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                          bias_regularizer=tf.keras.regularizers.l2(1e-4),
                          activity_regularizer=tf.keras.regularizers.l2(1e-5),
                          return_sequences=False,
                          return_state=False,
                          trainable=True)(embed_layer)
        #print(lstm_layer.shape)
        dense_layer = Dense(dense_size,
                            activation="relu",
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=tf.keras.regularizers.l2(1e-4),
                            activity_regularizer=tf.keras.regularizers.l2(1e-5),
                            trainable=True)(lstm_layer)
        drop_layer = Dropout(Params.drop_out)(dense_layer)
        #print(drop_layer.shape)
        output_layer = Dense(label_size,
                             activation="softmax",
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                             bias_regularizer=tf.keras.regularizers.l2(1e-4),
                             activity_regularizer=tf.keras.regularizers.l2(1e-5),
                             trainable=True)(drop_layer)
        #print(output_layer.shape)
        self.model = Model(input_layer, output_layer)

    def get_model(self):
        return self.model
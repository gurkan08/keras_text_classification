
import fasttext
import os

class Params(object):

    max_sent_size = None # mean sentence size in train dataset
    embedding_matrix = None
    embed_size = 300 # 300 for fasttext embeds
    lstm_units = 100
    dense_size = 50
    drop_out = 0.3

    epochs = 100
    batch_size = 1024 # 64
    lr = 0.00025
    test_size = 0.3
    validation_split = 0.1
    shuffle_count = 50
    early_stop_patience = 2

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fasttext_model = fasttext.load_model("model/cc.tr.300.bin")


from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import tensorflow as tf
import pandas as pd

from api_service.codes.preprocess import *

graph = tf.get_default_graph()

#model_dir = os.path.join(os.getcwd(), "api_service", "codes", "model") # local django server run
model_dir = os.path.join("text_classification_service", "api_service", "codes", "model") # docker django server run
max_sentence_size = 186
shuffle_count = 50
with open(os.path.join(model_dir, "sentence_tokenizer.pickle"), "rb") as handle:
    sentence_tokenizer = pickle.load(handle)
with open(os.path.join(model_dir, "label_tokenizer.pickle"), "rb") as handle:
    label_tokenizer = pickle.load(handle)
model = load_model(os.path.join(model_dir, "model.h5"))
print("----loaded model summary------")
print(model.summary())

def run_preprocess(data):
    data = lowercase(data)
    data = remove_punctuations(data)
    data = remove_numbers(data)
    data = remove_stop_words(data)
    # data = zemberek_stemming(data)
    data = first_5_char_stemming(data)
    data = data_shuffle(data, shuffle_count)
    return data

def api(text):
    text = " ".join(text.rstrip().lower().split())
    data = pd.DataFrame(zip([text], ["_dummy_label_"]), columns=["text", "label"])
    data = run_preprocess(data)

    out = sentence_tokenizer.texts_to_sequences(data["text"])  # list [] format
    out = pad_sequences(out, maxlen=max_sentence_size, padding="post", value=0.)

    with graph.as_default():
        pred = model.predict(out, verbose=1)
    result = {}
    for key, value in label_tokenizer.word_index.items():
        result[key] = pred[0][value - 1] # -1: label_tokenizer dicts starts from 1
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)) # sort by class prob
    return result

if __name__ == '__main__':

    text = """
    Sony cephesinden son gelen haber ise önümüzdeki hafta düzenlenecek olan etkinlik ile ilgili oldu. Sony, geçtiğimiz dönemde de PlayStation 5 ile ilgili bir etkinlik düzenlemişti. Ancak etkinlik süresince verilen bilgiler izleyenleri pek tatmin etmemişti. Bahsi geçen etkinlik boyunca yeni oyunları, cihazın bazı özellikleri ve tasarımını gösteren Sony, kullanıcıların gözünde pek verimli bir etkinlik düzenleyememişti.
    """
    result = api(text)
    print(result)


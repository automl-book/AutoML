import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from flask import Flask, request
import json

app=Flask(__name__)

CLASSES = {'CPU_Utilization': 0, 'Password_Reset': 1, 'Memory_Utilization': 2}

TOP_K = 25000
# Sentences will be truncated/padded to this length
MAX_LENGTH = 50

def return_data(df):
    return list(df['Text']), np.array(df['Label'].map(CLASSES))

def embedding_matrix_conv(index_word, embedding_path, embedding_dim):
    embedding_matrix_comb = {}
    with open(embedding_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix_comb[word] = coefs
    num_words = min(len(index_word) + 1, TOP_K)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in index_word.items():
        if i >= TOP_K:
            continue
        embedding_vector = embedding_matrix_comb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



@app.route ("/createModel", methods=['POST'])
def createModel():
    data = pd.read_csv('Dataset.csv', header = None)
    data.columns = ['Text', 'Label']
    train = data.sample(frac=0.8)
    data.drop(train.index, axis=0, inplace=True)
    # 10% for validation
    valid = data.sample(frac=0.5)
    data.drop(valid.index, axis=0, inplace=True)
    # 10% for test
    test = data

    # Apply it to the three splits
    train_text, train_labels = return_data(train)
    valid_text, valid_labels = return_data(valid)
    test_text, test_labels = return_data(test)

    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_text)

    filters=64
    dropout_rate=0.2
    embedding_dim=200
    kernel_size=3
    pool_size=3
    index_word=tokenizer.index_word
    embedding_path = 'glove.6B.200d.txt'
    embedding_dim=200

    def create_model():
      model = models.Sequential()
      num_features = min(len(index_word) + 1, TOP_K)
      # Add embedding layer - GloVe embeddings
      model.add(Embedding(input_dim=num_features,
                    output_dim=embedding_dim,
                    input_length=MAX_LENGTH,
                    weights=[embedding_matrix_conv(index_word,
                                    embedding_path, embedding_dim)],
                    trainable=True))

      model.add(Dropout(rate=dropout_rate))
      model.add(Conv1D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    bias_initializer='he_normal',
                    padding='same'))
      model.add(MaxPooling1D(pool_size=pool_size))
      model.add(Conv1D(filters=filters * 2,
                    kernel_size=kernel_size,
                    activation='relu',
                    bias_initializer='he_normal',
                    padding='same'))
      model.add(GlobalAveragePooling1D())
      model.add(Dropout(rate=dropout_rate))
      model.add(Dense(len(CLASSES), activation='softmax'))
      # Compile model with learning parameters.
      optimizer = tf.keras.optimizers.Adam(lr=0.001)
      model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
      return model

    train_process = tokenizer.texts_to_sequences(train_text)
    train_process = sequence.pad_sequences(train_process, maxlen=MAX_LENGTH)

    valid_process = tokenizer.texts_to_sequences(valid_text)
    valid_process = sequence.pad_sequences(valid_process, maxlen=MAX_LENGTH)

    test_process = tokenizer.texts_to_sequences(test_text)
    test_process = sequence.pad_sequences(test_process, maxlen=MAX_LENGTH)

    model = create_model()
    model.fit(train_process, train_labels, epochs=50)
    model.save('model_saved/model')
    
    return "Model Created Successfully"


@app.route ("/predictclass", methods=['POST'])
def predictClass():
    json_data = request.get_json(force=True)
    content= json_data["content"]
    text = list(content)
    result = {}

    tokenizer = text.Tokenizer(num_words=TOP_K)
    pred_process = tokenizer.texts_to_sequences(text)
    pred_process = sequence.pad_sequences(pred_process, maxlen=MAX_LENGTH)
    new_model = tf.keras.models.load_model('model_saved/model')
    prediction = int(new_model.predict_classes(pred_process))

    for key, value in CLASSES.items():
        if value==prediction:
            category=key
            result["class"] = category
    result = {"results": result}
    result = json.dumps(result)
    return result

@app.route ("/getHeartBeat", methods=['GET'])
def getHeartBeat():
    return "ok"

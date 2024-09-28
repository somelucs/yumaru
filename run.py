import pandas as pd
import numpy as np
from flask import Flask,render_template, request
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding,Flatten,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


def textto(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

# Pré-processamento de texto
def preproces(texto, token=None, lenn=None):
    if token is None:
        token = Tokenizer()
        token.fit_on_texts([texto])
    
    sequences = token.texts_to_sequences([texto])
    
    if lenn is None:
        lenn = max([len(sequence) for sequence in sequences])
    
    return sequences, token, lenn

# Criar as sequências de entrada e saída
def input_output(sequenc, lenn):
    input_sequenc = []
    for i in range(1, len(sequenc)):
        n_gram_sequenc = sequenc[:i+1]
        input_sequenc.append(n_gram_sequenc)

    input_sequenc = pad_sequences(input_sequenc, maxlen=lenn, padding='pre')
    X = input_sequenc[:, :-1]
    y = input_sequenc[:, -1]
    
    y = tf.keras.utils.to_categorical(y, num_classes=len(token.word_index)+1)
    
    return X, y

# Construção do modelo
def create_model(vocab, lenn):
    
    modelo = Sequential()
    modelo.add(Embedding(vocab, 50, input_length=lenn-1))
    modelo.add(LSTM(200, return_sequences=True))
    modelo.add(LSTM(100))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(vocab, activation='softmax'))
  
    modelo.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return modelo

def gen_text(seed, next_words, modelo, token, lenn):
    for _ in range(next_words):
        token_lista = token.texts_to_sequences([seed])[0]
        token_lista = pad_sequences([token_lista], maxlen=lenn-1, padding='pre')
        prev = np.argmax(modelo.predict(token_lista), axis=-1)
        
        output_w = ""
        for w, index in token.word_index.items():
            if index == prev:
                output_w = w
                break
        seed += " " + output_w
    return seed

fl = textto('D:/Py_Flask/nv/tst22.txt')

sequenc, token, lenn = preproces(fl)

X,y = input_output(sequenc, lenn)
# Criação e treinamento do modelo
modelo = create_model(len(token.word_index)+1, lenn)
modelo.fit(X, y, epochs=1000, verbose=0)

# Geração de texto
txt="how's it going"
gen = gen_text(txt, 1, modelo, token, lenn)


@app.route('/')
def index():
    return render_template('menu.html',ai=str(gen))


@app.route('/src', methods=["POST"])
def src():
    if request.method == 'POST':
        txt=request.form['txtb']
        gen = gen_text(txt, 1, modelo, token, lenn)

        return render_template('menu.html',ai=str(gen))


if __name__ == '__main__':
    app.run(debug=True)


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Dati di esempio
sentences = [
    "Questa è una frase positiva.",
    "Non mi piace affatto questa situazione.",
    "Adoro questa nuova canzone.",
    "Mi sento triste oggi.",
    "Questa giornata è fantastica!",
    "Ho ottenuto un risultato eccezionale al lavoro.",
    "Mi sento molto deluso per l'andamento delle cose.",
    "Ho perso il mio portafoglio e ora sono preoccupato.",
    "Incontro sempre persone straordinarie.",
    "La situazione economica mi preoccupa.",
    "La primavera porta sempre un sorriso sul mio viso.",
    "Sono grato per le piccole gioie della vita.",
    "Ogni nuvola ha un suo lato positivo, anche se non lo vediamo sempre.",
    "Il calore del sole mi riempie di energia positiva.",
    "Anche nelle giornate grigie, cerco la bellezza intorno a me.",
    "Le risate con gli amici rendono ogni giornata speciale.",
    "Ho superato una sfida personale, e ora mi sento invincibile.",
    "L'apprezzamento per le piccole cose rende la vita più ricca.",
    "Una tazza di caffè caldo può migliorare immediatamente il mio umore.",
    "Non importa quanto difficile sia il giorno, il sorriso di un amico può cambiarlo tutto.",
    "Le cose non stanno andando come avevo pianificato.",
    "Ogni giorno sembra portare nuovi problemi.",
    "Mi sento bloccato in una routine senza via d'uscita.",
    "Le aspettative spesso portano solo delusioni.",
    "Non riesco a trovare motivazione per affrontare la giornata.",
    "Le relazioni sono complicate, e mi sento frustrato.",
    "Le notizie negative sembrano sempre sovrastare quelle positive.",
    "I sogni sembrano così lontani dalla realtà attuale.",
    "La monotonia quotidiana sta lentamente consumando la mia felicità.",
    "Le delusioni sembrano essere all'ordine del giorno."
]

labels = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 per frasi positive, 0 per frasi negative

# Creazione del Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Sequenze di testo e padding
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Creazione del modello
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=8, input_length=20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilazione del modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(padded_sequences, labels, epochs=20, verbose=1)

# Test di nuove frasi

test_sentences = [
    "Questa giornata è fantastica!",
    "Mi sento giù oggi.",
    "Ogni nuvola ha un suo lato positivo, anche se non lo vediamo sempre.",
    "Le aspettative spesso portano solo delusioni.",
    "Le risate con gli amici rendono ogni giornata speciale.",
    "La monotonia quotidiana sta lentamente consumando la mia felicità."
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=20, padding='post', truncating='post')

# Predizione sulle nuove frasi
predictions = model.predict(padded_test_sequences)

# Mostra le frasi e le relative previsioni
for i in range(len(test_sentences)):
    print(f'Frase: {test_sentences[i]} - Predizione: {"Positiva" if predictions[i] > 0.5 else "Negativa"}')

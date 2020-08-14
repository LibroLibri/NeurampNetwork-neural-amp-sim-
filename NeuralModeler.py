import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from sys import argv, exit
from os import sep, listdir
from os.path import join
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2
EPOCHS = 70
CHUNK_SIZE = 1000

ACCURACY_TRESHOLD = 0.0001

from wav_processing import convert_to_array, split_into_chunks

USAGE = "USAGE: python NeuralModeler.py AmpInputFolder AmpOutputFolder ModelName"

def main():
    if len(argv) != 4:
        print(USAGE)
        exit(1)

    dataIN, dataOUT = load_data(argv[1], argv[2])
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataIN), np.array(dataOUT), test_size=TEST_SIZE
    )

    del dataIN
    del dataOUT

    model = get_model()

    model.fit(x_train, y_train, epochs=15)
    model.evaluate(x_test,  y_test, verbose=2)

    
    last_accuracy = model.evaluate(x_test,  y_test, verbose=2)[1]

    filename = argv[3]

    model.save(filename)

    model.fit(x_train, y_train, epochs=10)

    new_accuracy = model.evaluate(x_test,  y_test, verbose=2)[1]

    if last_accuracy > new_accuracy:
        print('WARNING: Trained for too many EPOCHS, accuracy is DECREASING')
        exit(0)
        
    last_accuracy = new_accuracy
    model.save(filename)

    while True:
            model.fit(x_train, y_train, epochs=10)

            new_accuracy = model.evaluate(x_test,  y_test, verbose=2)[1]

            if last_accuracy > new_accuracy:
                print('WARNING: Trained for too many EPOCHS, accuracy is DECREASING')
                break

            if new_accuracy - last_accuracy < ACCURACY_TRESHOLD:
                break

            last_accuracy = new_accuracy
            model.save(filename)
            
    



def load_data(x_dir, y_dir):
    dataIN = []
    dataOUT = []
    # Load Wav files
    for fileIN in listdir(x_dir):
        dataIN.append(convert_to_array(join(x_dir, fileIN)))
    for fileOUT in listdir(y_dir):
        dataOUT.append(convert_to_array(join(y_dir, fileOUT)))
    assert(len(dataIN) == len(dataOUT))
    # Split chunks
    chunks_in = []
    chunks_out = []
    for wav_array in dataIN:
        for array_chunk in split_into_chunks(wav_array, CHUNK_SIZE):
            chunks_in.append(array_chunk)
    for wav_array in dataOUT:
        for array_chunk in split_into_chunks(wav_array, CHUNK_SIZE):
            chunks_out.append(array_chunk)
    return (chunks_in, chunks_out)

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(CHUNK_SIZE,)),
        tf.keras.layers.Dense(300, activation='relu'),

        tf.keras.layers.Dense(300, activation='relu'),

        tf.keras.layers.Dense(300, activation='relu'),

        tf.keras.layers.Dense(300, activation='relu'),

        tf.keras.layers.Dense(CHUNK_SIZE, activation='tanh')
    ])
    model.compile(
        optimizer='adamax',
        loss='mean_absolute_error',
        metrics=['accuracy']
    )
    return model
    
if __name__ == "__main__":
    main()

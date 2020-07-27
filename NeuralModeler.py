import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from sys import argv
from os import sep, listdir
from os.path import join

TEST_SIZE = 0.3

from wav_processing import convert_to_array, convert_to_wav, split_into_chunks

USAGE = "python NeuralModeler.py AmpInputFolder/ AmpOutputFolder/ ModelName"

def main():
    if len(argv) != 4:
        print(USAGE)
    data = load_data(argv[1], argv[2])
    test_index = int(len(data[0]) * TEST_SIZE)
    x_train, y_train, x_test, y_test = data[0][test_index:], data[1][test_index:], data[0][:test_index], data[1][:test_index]
    model = get_model()


def load_data(x_dir, y_dir):
    dataIN = []
    dataOUT = []
    # Load Wav files
    for fileIN in listdir(x_dir):
        dataIN.append(convert_to_array(join(x_dir, fileIN)))
    for fileOUT in listdir(y_dir):
        dataOUT.append(convert_to_array(join(y_dir, fileOUT))[1])
    assert(len(dataIN) == len(dataOUT))
    # Split chunks
    chunks_in = []
    chunks_out = []
    for wav_array in dataIN:
        for array_chunk in split_into_chunks(wav_array, 10000):
            chunks_in.append(array_chunk)
    for wav_array in dataOUT:
        for array_chunk in split_into_chunks(wav_array, 10000):
            chunks_out.append(array_chunk)
    return (chunks_in, chunks_out)

def get_model():
    pass

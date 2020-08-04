import tensorflow as tf
from sys import argv, exit
import numpy as np
from wav_processing import convert_to_array, convert_to_wav, split_into_chunks

CHUNK_SIZE = 1000

USAGE = "USAGE: python neuramp.py InFile.wav OutFile.wav amp-model"

def main():
    if len(argv) != 4:
        print(USAGE)
        exit(1)
    model = tf.keras.models.load_model(argv[3])
    data = convert_to_array(argv[1])
    output = model.predict(np.array(split_into_chunks(data, chunk_size=CHUNK_SIZE)))
    convert_to_wav(argv[2], output)



if __name__ == '__main__':
    main()

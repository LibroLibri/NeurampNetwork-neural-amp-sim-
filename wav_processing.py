import numpy as np
from scipy.io import wavfile
from sys import exit

WAV_format_ranges = {
    np.dtype('float32'): [-1.0, 1.0],
    np.dtype('int32'): [-2147483648, 2147483647],
    np.dtype('int16'): [-32768, 32767],
    np.dtype('uint8'): [0, 255]
}



# TODO: resample if not 44.1 KHz

def convert_to_array(filename, target_datatype='float32', target_samplerate=44100):
    # Read file
    samplerate, data = wavfile.read(filename)
    source_range = WAV_format_ranges[data.dtype]
    # Check that the file format is valid
    if data.dtype not in WAV_format_ranges:
        raise InputError(filename, 'File Format not supported. Try with 8, 16, 32 or 32-Float formats')
    if samplerate != target_samplerate:
        raise InputError(filename, 'Samplerate not supported. Use files with 44.1 KHz samplerate')
    # Stereo to Mono
    if len(data.shape) != 1:
        print('Converting Stereo to Mono')
        data = data.sum(axis=1) / 1.5
    # Take file to float32 format
    if data.dtype != np.dtype(target_datatype):
        print('changing format')
        data = data.astype(target_datatype)
        data = np.interp(data, source_range, WAV_format_ranges[np.dtype(target_datatype)])
    return data
    
    

def convert_to_wav(filename, data, samplerate=44100, target_datatype='float32', source_range=[-1.0, 1.0]):
    # Take data to target range and datatype
    if WAV_format_ranges[target_datatype] != source_range:
        data = np.interp(data, source_range, WAV_format_ranges[np.dtype(target_datatype)])
    data = data.astype(target_datatype)
    # Write file and return
    wavfile.write(filename, samplerate, data)
    return data


def split_into_chunks(data, chunk_size=22050):
    # Calculate how many chunks are necesary
    n_chunks = int(chunk_size / float(data.shape[0]) - (chunk_size / float(data.shape[0])) % 1)
    if chunk_size % data.shape[0] != 0:
        n_chunks += 1
    # Create chunks filled with 0s
    chunks = []
    # Put data in the chunks
    while True: 
        if data.shape[0] != 0:
            chunks.append(data[:chunk_size])
            data = data[chunk_size:]
        else:
            break
    while chunks[-1].shape[0] != chunk_size:
        chunks[-1] = np.append(chunks[-1], (0))
    return chunks

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

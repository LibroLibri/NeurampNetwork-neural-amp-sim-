import numpy as np
from scipy.io import wavfile
from sys import exit

WAV_format_ranges = {
    np.dtype('float32'): [-1.0, 1.0],
    np.dtype('int32'): [-2147483648, 2147483647],
    np.dtype('int16'): [-32768, 32767],
    np.dtype('uint8'): [0, 255]
}


# TODO: remove right channel if stereo
# TODO: resample if not 44.1 KHz


def convert_to_array(filename, target_datatype, target_samplerate):
    # Read file
    samplerate, data = wavfile.read(filename)
    source_range = WAV_format_ranges[data.dtype]
    # Check that the file format is valid
    if data.dtype not in WAV_format_ranges:
        raise InputError(filename, 'File Format not supported. Try with 8, 16, 32 or 32-Float formats')
    if samplerate != 44100:
        raise InputError(filename, 'Samplerate not supported. Use files with 44.1 KHz samplerate')
    # Stereo to Mono
    if data.shape[1] != 1:
        print('Converting Stereo to Mono')
        data = data.sum(axis=1) / 1.5
    # Take file to float32 format
    if data.dtype != np.dtype(target_datatype):
        print('changing format')
        data = data.astype(target_datatype)
        data = np.interp(data, source_range, WAV_format_ranges[np.dtype(target_datatype)])
    return data
    
    

def convert_to_wav(filename, data, target_datatype):
    
    pass
class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

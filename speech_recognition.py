'''
Moved the model retraining from ipynb to py due to Jupyter issues
'''

import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex # general numpy complex fix

import os
import glob
import random
import shutil

import librosa
import soundfile as sf
# from IPython.display import Audio
# IPython only works on Jupyter, thus removed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
from tflite_model_maker.config import ExportFormat

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")
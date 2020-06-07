import os
import json
import string
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from functools import reduce
from random import random
from PIL import Image
from time import time
import matplotlib.pyplot as plt

import tensorflow.keras.applications.inception_v3
import tensorflow.keras.preprocessing.image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical 
from loader import Loader
from model import Model

def generateCaption(photo, model):
    in_text = loader.START
    for _ in range(loader.max_length):
        sequence = [loader.wordtoidx[w] for w in in_text.split() if w in loader.wordtoidx]
        sequence = pad_sequences([sequence], maxlen=loader.max_length)
        yhat = model.caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = loader.idxtoword[yhat]
        in_text += ' ' + word
        print(word)
        if word == loader.STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

if __name__ == '__main__':

  loader = Loader()
  model = Model()
  
  loader.load(model)

  model.build(loader)
  model.train()

  # generate captions
  for img in loader.train_descriptions.keys():
      image = loader.encoding_train[f'{img}.jpg'].reshape((1,model.OUTPUT_DIM))
      x = plt.imread(os.path.join(loader.root_path,'ImgFlip500K_Dataset','templates','img', f'{img}.jpg'))
      plt.imshow(x)
      plt.show()
      print(image)
      print("Caption:",generateCaption(image, model))
      print("_____________________________________")
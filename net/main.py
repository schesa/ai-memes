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
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print('Note: using Google CoLab')
    COLAB = True
except:
    print('Note: not using Google Colab')
    COLAB = False

if COLAB:
    root_path = "/content/drive/Shared drives/ai-memes"
else:
    root_path = "./data/"


def clean_data(list):
    null_punct = str.maketrans('', '', string.punctuation)
    for desc in list:
        desc = [word.lower() for word in desc.split()]
        desc = [w.translate(null_punct) for w in desc]
        desc = [word for word in desc if word.isalpha()]
        desc = [word for word in desc if len(word) > 1]
        yield ' '.join(desc)


def isascii_word(w): return len(w) == len(w.encode())


def isascii_list(l): return reduce(lambda rez, word: rez &
                                   True if isascii_word(word) else False & rez, l, True)


dirname = os.path.join(root_path, 'ImgFlip500K_Dataset', 'memes')
lookup = dict()


def print_iterator(it):
    for x in it:
        print(x, end='\n')
    print('')  # for new line


def load_memes(start=0, end=100):
    i = 0
    for filename in tqdm(os.listdir(dirname)):  # foreach json file in memes
        i += 1
        if i < start or i > end:
            continue
        meme_name = Path(filename).stem  # remove file extension
        with open(os.path.join(dirname, filename)) as json_file:
            memes = json.load(json_file)
            # lookup[meme_name] = list(map(lambda meme: ' | '.join(meme['boxes']), memes))
            lookup[meme_name] = []
            for meme in memes:
                words = ' | '.join(clean_data(meme['boxes']))
                if isascii_list(words):
                    lookup[meme_name].append(words)


load_memes()
print(f'Memes loaded: {len(lookup)}')  # 99 memes, in the latest dataset 100
print(f'Meme example: {lookup[list(lookup)[0]][0]}') # when you off the dope | and you think you a bird

# load unique words
max_length = 0
lex = set()  # set of unique words
for desc in lookup:
    for word in desc.split():
        lex.add(word)
        if max_length < len(word):
            max_length = max(max_length, len(word))
            max_word = word

print(f'Unique words: {len(lex)}')
print(f'Biggest word: {max_length} chars')
print(max_word)

# same as lookup but captions wrapped in Start/Stop
START = "startseq"
STOP = "endseq"
train_descriptions = {k:v for k,v in lookup.items()}
for n,v in train_descriptions.items(): 
  for d in range(len(v)):
    v[d] = f'{START} {v[d]} {STOP}'
max_length += 2 # update maximum caption length
print(f'Wrapped captions in Start/Stop')

# load CNN
encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
print('InceptionV3 loaded!')

def encodeImage(img):
  # Resize all images to a standard size (specified bythe image encoding network)
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
  # Convert a PIL image to a numpy array
  x = tensorflow.keras.preprocessing.image.img_to_array(img)
  # Expand to 2D array
  x = np.expand_dims(x, axis=0)
  # Perform any preprocessing needed by InceptionV3 or others
  x = preprocess_input(x)
  # Call InceptionV3 (or other) to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image
  # Shape to correct form to be accepted by LSTM captioning network.
  x = np.reshape(x, OUTPUT_DIM )
  return x

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

# generate train set
train_path = os.path.join(root_path,'ImgFlip500K_Dataset',"data",f'train{OUTPUT_DIM}.pkl')
if not os.path.exists(train_path):
  start = time()
  encoding_train = {}
  train_images_path = os.path.join(root_path,'ImgFlip500K_Dataset','templates','img') 
  for image_path in tqdm(os.listdir(train_images_path)):
    # print(image_path)
    img = tensorflow.keras.preprocessing.image.load_img(os.path.join(train_images_path,image_path), target_size=(HEIGHT, WIDTH))
    encoding_train[image_path] = encodeImage(img)
  with open(train_path, "wb") as fp:
    pickle.dump(encoding_train, fp)
  print(f"\nGenerating training set took: {hms_string(time()-start)}")
else:
  with open(train_path, "rb") as fp:
    encoding_train = pickle.load(fp)
print('Loaded')

# build all captions list
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

# remove words with occurences < 1
word_count_threshold = 1
word_counts = {}
nsents = 0 # number of sentences
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

# idxtoword and wordtoidx
idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1
    
vocab_size = len(idxtoword) + 1 
vocab_size

#load glove
glove_dir = os.path.join(root_path,'glove.6B')
embeddings_index = {} 
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print(f'Found {len(embeddings_index)} word vectors.')

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
not_found = []
for word, i in wordtoidx.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
    else: 
      not_found.append(word)
print(f'Not founded words: {len(not_found)} from total of {len(wordtoidx)}')

# delete wrong sentences
eliminated = 0
def hasNotFoundWord(desc):
  global eliminated
  stripped_desc = desc.split()[1:-1]
  if bool(set(not_found) & set(stripped_desc)):
    eliminated += 1
    return True
  else:
    return False

for meme in tqdm(train_descriptions):
    train_descriptions[meme] = [desc for desc in train_descriptions[meme] if hasNotFoundWord(desc) == False ]
print(f"\nGenerated Training Descriptions")

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
# print(f'Removed captions: {eliminated} from total of {len(all_train_captions)}')

# build Network
print('Building model')
inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2,se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1,inputs2], outputs=outputs)
caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
print('Compiling model')
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# load weights
model_path = os.path.join(root_path,'ImgFlip500K_Dataset',"data",'caption-model.hdf5')
if not os.path.exists(model_path):
  print('Can not train on this machine')
else:
  caption_model.load_weights(model_path)
  print('Model loaded')

def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        print(word)
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
# generate captions
for img in train_descriptions.keys():
    image = encoding_train[f'{img}.jpg'].reshape((1,OUTPUT_DIM))
    x = plt.imread(os.path.join(root_path,'ImgFlip500K_Dataset','templates','img', f'{img}.jpg'))
    plt.imshow(x)
    plt.show()
    print(image)
    print("Caption:",generateCaption(image))
    print("_____________________________________")
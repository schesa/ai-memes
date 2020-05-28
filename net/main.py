import os
import string
import glob
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3

from tqdm import tqdm
import tensorflow.keras.preprocessing.image
import pickle
from time import time
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import numpy as np

# Nicely formatted time string
START = "startseq"
STOP = "endseq"
EPOCHS = 10

root_captioning = "/content/drive/My Drive/licenta AC/Chesa/data"


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


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
    x = encode_model.predict(x)  # Get the encoding vector for the image
    # Shape to correct form to be accepted by LSTM captioning network.
    x = np.reshape(x, OUTPUT_DIM)
    return x


def data_generator(descriptions, photos, wordtoidx, max_length, num_photos_per_batch):
    # x1 - Training data for photos
    # x2 - The caption that goes with each photo
    # y - The predicted rest of the caption
    x1, x2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            n += 1
            photo = photos[key + '.jpg']
            # Each photo has 5 descriptions
            for desc in desc_list:
                # Convert each word into a list of sequences.
                seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
                # Generate a training case for every possible sequence and outcome
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    x1.append(photo)
                    x2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield ([np.array(x1), np.array(x2)], np.array(y))
                x1, x2, y = [], [], []
                n = 0


def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


if __name__ == "__main__":
    null_punct = str.maketrans('', '', string.punctuation)
    lookup = dict()

    with open(os.path.join(root_captioning, 'Flickr8k_text', 'Flickr8k.token.txt'), 'r') as fp:
        max_length = 0
        for line in fp.read().split('\n'):
            tok = line.split()
            if len(line) >= 2:
                id = tok[0].split('.')[0]
                desc = tok[1:]

                # Cleanup description
                desc = [word.lower() for word in desc]
                # desc = [w.translate(null_punct) for w in desc]
                # desc = [word for word in desc if len(word)>1]
                # desc = [word for word in desc if word.isalpha()]
                max_length = max(max_length, len(desc))

                if id not in lookup:
                    lookup[id] = list()
                lookup[id].append(' '.join(desc))

    lex = set()
    for key in lookup:
        [lex.update(d.split()) for d in lookup[key]]

    # Warning, running this too soon on GDrive can sometimes not work.
    # Just rerun if len(img) = 0
    img = glob.glob(os.path.join(root_captioning, 'Flicker8k_Dataset', '*.jpg'))

    # Read all image names and use the predefined train/test sets.
    train_images_path = os.path.join(root_captioning, 'Flickr8k_text', 'Flickr_8k.trainImages.txt')
    train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
    test_images_path = os.path.join(root_captioning, 'Flickr8k_text', 'Flickr_8k.testImages.txt')
    test_images = set(open(test_images_path, 'r').read().strip().split('\n'))

    train_img = []
    test_img = []

    for i in img:
        f = os.path.split(i)[-1]
        if f in train_images:
            train_img.append(f)
        elif f in test_images:
            test_img.append(f)

    # Build the sequences.  We include a **start** and **stop** token at the beginning/end.
    # We will later use the **start** token to begin the process of generating a caption.
    # Encountering the **stop** token in the generated text will let us know we are done.
    train_descriptions = {k: v for k, v in lookup.items() if f'{k}.jpg' in train_images}
    for n, v in train_descriptions.items():
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    # CNN
    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)
    WIDTH = 299
    HEIGHT = 299
    OUTPUT_DIM = 2048
    preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

    # Generate training set
    train_path = os.path.join(root_captioning, "data", f'train{OUTPUT_DIM}.pkl')
    if not os.path.exists(train_path):
        start = time()
        encoding_train = {}
        for id in tqdm(train_img):
            image_path = os.path.join(root_captioning, 'Flicker8k_Dataset', id)
            img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
            encoding_train[id] = encodeImage(img)
        with open(train_path, "wb") as fp:
            pickle.dump(encoding_train, fp)
        print(f"\nGenerating training set took: {hms_string(time() - start)}")
    else:
        with open(train_path, "rb") as fp:
            encoding_train = pickle.load(fp)

    # Generate test set
    test_path = os.path.join(root_captioning, "data", f'test{OUTPUT_DIM}.pkl')
    if not os.path.exists(test_path):
        start = time()
        encoding_test = {}
        for id in tqdm(test_img):
            image_path = os.path.join(root_captioning, 'Flicker8k_Dataset', id)
            img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
            encoding_test[id] = encodeImage(img)
        with open(test_path, "wb") as fp:
            pickle.dump(encoding_test, fp)
        print(f"\nGenerating testing set took: {hms_string(time() - start)}")
    else:
        with open(test_path, "rb") as fp:
            encoding_test = pickle.load(fp)

    # Separate captions
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    len(all_train_captions)

    # Remove words < 10 occurences
    # TODO skip this step
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

    # Build lookup tables
    idxtoword = {}
    wordtoidx = {}

    ix = 1
    for w in vocab:
        wordtoidx[w] = ix
        idxtoword[ix] = w
        ix += 1

    vocab_size = len(idxtoword) + 1
    vocab_size

    # For Start Stop Tokens
    max_length += 2

    # Load Glove Embeddings
    glove_dir = os.path.join(root_captioning, 'glove.6B')
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

    for word, i in wordtoidx.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)

    inputs1 = Input(shape=(OUTPUT_DIM,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    caption_model.layers[2].set_weights([embedding_matrix])
    caption_model.layers[2].trainable = False
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train
    number_pics_per_bath = 3
    steps = len(train_descriptions) // number_pics_per_bath
    model_path = os.path.join(root_captioning, "data", f'caption-model.hdf5')
    if not os.path.exists(model_path):
        for i in tqdm(range(EPOCHS * 2)):
            generator = data_generator(train_descriptions, encoding_train, wordtoidx, max_length, number_pics_per_bath)
            caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

        caption_model.optimizer.lr = 1e-4
        number_pics_per_bath = 6
        steps = len(train_descriptions) // number_pics_per_bath

        for i in range(EPOCHS):
            generator = data_generator(train_descriptions, encoding_train, wordtoidx, max_length, number_pics_per_bath)
            caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        caption_model.save_weights(model_path)
        print(f"\Training took: {hms_string(time() - start)}")
    else:
        caption_model.load_weights(model_path)

    # Evaluate Performance
    for z in range(10):
        pic = list(encoding_test.keys())[z]
        image = encoding_test[pic].reshape((1, OUTPUT_DIM))
        print(os.path.join(root_captioning, 'Flicker8k_Dataset', pic))
        x = plt.imread(os.path.join(root_captioning, 'Flicker8k_Dataset', pic))
        plt.imshow(x)
        plt.show()
        print("Caption:", generateCaption(image))
        print("_____________________________________")

    # Evaluate My photos

    urls = [
        "https://scontent.ftsr1-1.fna.fbcdn.net/v/t31.0-8/20645317_1443234962397297_4448855943174430549_o.jpg?_nc_cat=101&_nc_ohc=WH_W291FOzsAX8XjRWn&_nc_ht=scontent.ftsr1-1.fna&oh=eed27ca12942acf634b4b178d7cd0a78&oe=5EEDD1AB",
        "https://akm-img-a-in.tosshub.com/indiatoday/images/story/201802/Marital_rape.jpeg?89BCnTS_y59u07KFC3CXxs.E02VE3ul7",
        "https://www.gcvs.com/wp-content/uploads/2019/06/why_dog_eats_grass.jpg",
        "https://acegif.com/wp-content/uploads/funny-dog-108-gap.jpg",
        "https://scontent.ftsr1-2.fna.fbcdn.net/v/t1.0-9/p720x720/86465661_3124449510919679_578305728519864320_o.jpg?_nc_cat=103&_nc_ohc=uQ_mgL1zeRwAX8tRej3&_nc_ht=scontent.ftsr1-2.fna&_nc_tp=6&oh=a61ecc8bd653e4510a56363df2a94933&oe=5EC39F0B",
        "https://scontent.ftsr1-2.fna.fbcdn.net/v/t31.0-8/p960x960/18238183_1546031648761481_7059126586853451427_o.jpg?_nc_cat=109&_nc_ohc=7JQDsxYCqmIAX_QVb35&_nc_ht=scontent.ftsr1-2.fna&_nc_tp=6&oh=126c762c2f8249cfb29f5ec127f309de&oe=5EEEF2D8",
        "https://scontent.ftsr1-2.fna.fbcdn.net/v/t31.0-8/14125677_1245451642152818_4851410731831734119_o.jpg?_nc_cat=105&_nc_ohc=4NT-oWNVUgQAX9HPseZ&_nc_ht=scontent.ftsr1-2.fna&oh=0138f1531ec52423540e44bc91e10f1b&oe=5EED99B3"
    ]

    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.load()

        plt.imshow(img)
        plt.show()

        response = requests.get(url)

        img = encodeImage(img).reshape((1, OUTPUT_DIM))
        print(img.shape)
        print("Caption:", generateCaption(img))
        print("_____________________________________")
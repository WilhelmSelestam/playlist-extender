
# Data loading
# streaming
# shuffeling

# Tokenization
# songs -> numbers

# Embedding
# tokens -> vectors?, matrices?, tensors?, instantiated as noise, learned during training
# position

# Attention?????
# idk
# multi headed?

# Transformer?????

# Training
# backpropogation?

from SongTokenizer import SongTokenizer
import json
import glob
import os


def stream_dataset(path):
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice["playlists"]:
              yield playlist


DATA_DIR = "C:/Users/seles/playlist-extender/dataset/data"
VOCAB_FILE = "./vocab.json"



# dataset_generator = stream_dataset(DATA_DIR)
# tokenizer = SongTokenizer(min_freq=20)
# tokenizer.fit(dataset_generator)
# tokenizer.save(VOCAB_FILE)


f = open('./dataset/data/mpd.slice.0-999.json')
js = f.read()
f.close()
data = json.loads(js)
  
data = data['playlists']


tokenizer = SongTokenizer()
tokenizer.load(VOCAB_FILE)

input_sequence = tokenizer.encode(data[0], max_length=60)

print(f"Encoded Sequence: {input_sequence}")
print(tokenizer.length())


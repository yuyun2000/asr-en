
import tensorflow as tf
import cv2
from prepro import compute_log_mel_fbank
import librosa
import numpy as np
from prepro import compute_log_mel_fbank_fromsig

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("./data/500hbpe-500.json")

f,rate = librosa.load('./today.mp3',sr=16000)
f = f*32768
f = f.astype(np.int16)
out = compute_log_mel_fbank_fromsig(f,rate)
print(out.shape)

model = tf.keras.models.load_model("./h5/en-0.h5")

if out.shape[0]<1600:
    img = np.pad(out,((0,1600-out.shape[0]),(0,0)))
else:
    img = cv2.resize(out, (80, 1600), interpolation=cv2.INTER_AREA).astype(np.float32)
# img = img / 255
print(img.shape)
img = img.reshape((1, 1600, 80, 1))

out = model(img)

out = np.argmax(out, axis=-1).reshape(200)
print(out)
text = tokenizer.decode(out)

print(text)


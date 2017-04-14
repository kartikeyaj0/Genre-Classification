import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt

sample_rate, X = scipy.io.wavfile.read("blues.00000.wav")
print(sample_rate, X.shape)
plt.specgram(X, Fs=sample_rate, xextent=(0,30))
plt.show()
import os
import scipy
def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    scipy.save(data_fn, fft_features)

with open('abc') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for x in content:
	create_fft(x)


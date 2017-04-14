import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
from scikits.talkbox.features import mfcc

def write_ceps(ceps, fn):
	base_fn, ext = os.path.splitext(fn)
	data_fn = base_fn + ".ceps"
	np.save(data_fn, ceps)
	print("Written to %s" % data_fn)

def create_ceps(fn):
	sample_rate, X = scipy.io.wavfile.read(fn)
	ceps, mspec, spec = mfcc(X)
	write_ceps(ceps, fn)

with open('abc') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for x in content:
	create_ceps(x)
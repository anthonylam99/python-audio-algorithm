import numpy as np
import scipy as sp
import scipy.io.wavfile

fname = 'steg_brilliant.wav'

rate, channels = sp.io.wavfile.read(fname)
channels = channels.copy()

msglen = 8 * 21
seglen = 2*int(2**np.ceil(np.log2(2*msglen)))       #lay gia tri do dai 1 segment
segmid = seglen // 2                                # gia tri goc cua segment

#lay gia tri khoi segemnts dau tien
if len(channels.shape) == 1:
    x = channels[:seglen]              #bien doi data audio sang dang int 
else:
    x = channels[:seglen,0]



x = (np.angle(np.fft.fft(x))[segmid-msglen:segmid] < 0).astype(np.int8)         #bien doi data sang dang int
x = x.reshape((-1,8)).dot(1 << np.arange(8 - 1, -1, -1))


print("Message: \n" + ''.join(np.char.mod('%c',x)))




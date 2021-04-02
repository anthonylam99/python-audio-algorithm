import numpy as np
import scipy as sp
import scipy.io.wavfile

fname = 'brilliant.wav'

rate, channels = sp.io.wavfile.read(fname)
channels = channels.copy()
# rate, channels.shape

msg = "this is sercret text"
msglen = 8 * len(msg)  #lay do dai bit cua message

seglen = int(2 * 2**np.ceil(np.log2(2*msglen)))  #do dai cua 1 segment (tinh theo so bit)
segnum = int(np.ceil(channels.shape[0]/seglen))  #so segment co trong channel audio


if len(channels.shape) == 1:
    channels.resize(segnum*seglen, refcheck=False)
    channels = channels[np.newaxis]
else:
    channels.resize((segnum*seglen, channels.shape[1]), refcheck=False)
    channels = channels.T


msgbin = np.ravel([[int(y) for y in format(ord(x), '08b')] for x in msg]) #chuyen doi chuoi msg thanh nhi phan

# ap dung thuat toan 
# neu data[i] = 0 => pi/2, data[i] = 1 => -pi/2
msgPi = msgbin.copy()
msgPi[msgPi == 0] = -1
msgPi = msgPi * -np.pi/2

#tra lai gia tri khoi cho kenh
segs = channels[0].reshape((segnum,seglen))

# bien doi tin hieu Fourier, Phase angles of first segment
segs = np.fft.fft(segs)

#du lieu dang DFT 
M = np.abs(segs)        #header
P = np.angle(segs)      # tinh toan do lech cua cac gia tri

# tinh bien do cua P
dP = np.diff(P, axis=0)

segmid = seglen // 2
P[0,-msglen+segmid:segmid] = msgPi                  #thay gia tri cua cac phan tu vao pha dau tien
P[0,segmid+1:segmid+1+msglen] = -msgPi[::-1]        #duy tri tinh doi xung cua DFT, lap lai tien trinh


for i in range(1, len(P)): P[i] = P[i-1] + dP[i-1]  #tinh toan lai do lech pha

# sau khi bien doi ta nhan duoc 1 chuoi data, sau do dem cong lai voi chuoi header M
segs = (M * np.exp(1j * P))

#dung lai tin hieu FFT theo khoi segs vua nhan duoc
segs = np.fft.ifft(segs).real

#tra gia tri ve dang 16 bit
channels[0] = segs.ravel().astype(np.int16)

#tao file ket qua
sp.io.wavfile.write('steg_'+fname, rate, channels.T)

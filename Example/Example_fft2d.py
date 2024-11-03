import nufft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform

numPix = 128
numDim = 2

img = transform.resize(data.shepp_logan_phantom(), [numPix for _ in range(numDim)]).astype(complex128)
# img *= exp(1j*pi/16)
if numDim == 3: img = tile(img[newaxis,:,:], (numPix,1,1))

ksp = fftshift(fftn(fftshift(img)))/(numPix**numDim)
arrK = array(meshgrid(
    *(linspace(-0.5, 0.5, numPix, endpoint=False) for _ in range(numDim)),
    indexing='ij')).reshape(numDim,-1).T
_img = nufft.ifft(ksp.flatten(), arrK, array([numPix for _ in range(numDim)])).reshape([numPix for _ in range(numDim)])

figure()
subplot(131)
imshow(real(img), cmap='gray')
subplot(132)
imshow(imag(img), cmap='gray')
subplot(133)
imshow(abs(img), cmap='gray')

figure()
subplot(131)
imshow(real(_img), cmap='gray')
subplot(132)
imshow(imag(_img), cmap='gray')
subplot(133)
imshow(abs(_img), cmap='gray')

show()

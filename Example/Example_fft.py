import nufft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform

numPix = 128
numDim = 3
kz=numPix//2; ky=numPix//2; kx=numPix//2

img = transform.resize(data.shepp_logan_phantom(), [numPix for _ in range(numDim)]).astype(complex128)
if numDim == 3: img = tile(img[newaxis,:,:], (numPix,1,1))

ksp = fftshift(fftn(fftshift(img)))/(numPix**numDim)
arrK = array(meshgrid(
    *(linspace(-0.5, 0.5, numPix, endpoint=False) for _ in range(numDim)),
    indexing='ij')).reshape(numDim,-1).T
_img = nufft.ifft(ksp.flatten(), arrK, array([numPix for _ in range(numDim)])).reshape([numPix for _ in range(numDim)])

figure()
subplot(231)
imshow(real(img[kz,:,:]), cmap='gray')
subplot(232)
imshow(real(img[:,ky,:]), cmap='gray')
subplot(233)
imshow(real(img[:,:,kx]), cmap='gray')
subplot(234)
imshow(imag(img[kz,:,:]), cmap='gray')
subplot(235)
imshow(imag(img[:,ky,:]), cmap='gray')
subplot(236)
imshow(imag(img[:,:,kx]), cmap='gray')

figure()
subplot(231)
imshow(real(_img[kz,:,:]), cmap='gray', vmin=0, vmax=1)
subplot(232)
imshow(real(_img[:,ky,:]), cmap='gray', vmin=0, vmax=1)
subplot(233)
imshow(real(_img[:,:,kx]), cmap='gray', vmin=0, vmax=1)
subplot(234)
imshow(imag(_img[kz,:,:]), cmap='gray', vmin=0, vmax=1)
subplot(235)
imshow(imag(_img[:,ky,:]), cmap='gray', vmin=0, vmax=1)
subplot(236)
imshow(imag(_img[:,:,kx]), cmap='gray', vmin=0, vmax=1)

show()

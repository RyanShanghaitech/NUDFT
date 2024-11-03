import nufft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform

numPix = 32
numDim = 3

img = transform.resize(data.shepp_logan_phantom(), (numPix,numPix)).astype(complex128)
if numDim == 3: img = tile(img[newaxis,:,:], (numPix,1,1))
ksp = fftshift(fftn(fftshift(img)))

lstCart_Np = array(meshgrid(
    *(linspace(-numPix//2, numPix//2, numPix, endpoint=False) for _ in range(numDim)),
    indexing='ij')).reshape(numDim,-1).T
lstCart_05 = array(meshgrid(
    *(linspace(-0.5, 0.5, numPix, endpoint=False) for _ in range(numDim)),
    indexing='ij')).reshape(numDim,-1).T

print(lstCart_Np.shape)
print(lstCart_05.shape)

ksp = nufft.dft(img.flatten(), lstCart_Np, lstCart_05)

ksp = ksp.reshape(*(numPix for _ in range(numDim)))
_img = fftshift(ifftn(fftshift(ksp)))

figure()
subplot(121)
if numDim == 3:
    imshow(abs(img[numPix//2,:,:]), cmap='gray')
else:
    imshow(abs(img), cmap='gray')
title("Original Image")
subplot(122)
if numDim == 3:
    imshow(abs(_img[numPix//2,:,:]), cmap='gray')
else:
    imshow(abs(_img), cmap='gray')
title("NUFFT Reconstruction")

show()
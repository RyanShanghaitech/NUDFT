# This example demonstrates how to use the NUDFT client to perform non-uniform discrete Fourier transform (NUDFT) with Cartesian data on a 2D image. To use non-Cartesian data, you can modify the input coordinates and corresponding data.

from numpy import *
from matplotlib.pyplot import *
import skimage.data as data
import skimage.transform as transform
from nudft import *

sizImg = 128
objClient = NudftClient()

def cfft(img): return fft.fftshift(fft.fft2(fft.ifftshift(img)))
def cifft(kspace): return fft.fftshift(fft.ifft2(fft.ifftshift(kspace)))

# generate phantom
img = data.shepp_logan_phantom()
img = transform.resize(img, [sizImg, sizImg])
kspace = cfft(img)

#
lstKxKy = loadtxt("./Resource/trjRadial.txt") # load sampling coordinates in k-space
lstDs = loadtxt("./Resource/lstDs_Radial.txt") # load sampling density compensation coefficient
lstXY = loadtxt("./Resource/trjCart.txt") # load sampling coordinates in image

# generate list of input (kspace) data
lstRawdata = objClient.nudft(img.flatten(), lstXY, lstKxKy)

# run NUIDFT
lstOutputData = objClient.nuidft(lstRawdata*lstDs, lstKxKy, lstXY)
imgReco = lstOutputData.reshape([sizImg, sizImg])

# show results
figure()
subplot(131)
imshow(abs(img), cmap="gray", vmin=0, vmax=1)
title("Img"); colorbar()
subplot(132)
plot(lstKxKy[:,0], lstKxKy[:,1], marker=".")
title("Sample in kspace"); axis("equal")
subplot(133)
imshow(abs(imgReco), cmap="gray")
title("Img->DFT->IDFT"); colorbar()

show()
import nudft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform

nPix = 256

arrM0 = transform.resize(data.shepp_logan_phantom(), (nPix,nPix)).astype(complex128)

# derive Cartesian coord, load Spiral coord, Aera array
tupK_Ct = meshgrid(
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    indexing='ij')[::-1]
arrK_Ct = array(tupK_Ct).transpose(1,2,0)
arrK_Sp = asarray(load("../Resource/K.npy"))
arrAera = asarray(load("../Resource/Aera.npy"))
nPE, nRO, _ = arrK_Sp.shape

# simulate image with offres effect
arrOm = zeros([nPix,nPix], dtype=complex128)
arrOm[:nPix//2,:] = (3.5e-6)*(2*pi)*(42.58e6)*(3) # fat
arrM0_Offres = array([arrM0*exp(1j*t*arrOm) for t in 2.5e-6*arange(nRO)])

# simulate rawdata S using dft()
arrS = zeros([nPE,nRO], dtype=complex128)
for iPE in range(nPE):
    arrS[iPE,:] = nudft.dft(arrM0_Offres.reshape(nRO,-1), arrK_Ct.reshape(-1,2), arrK_Sp[iPE,:,:])

# reconstruct image using idft()
imgReco = nudft.idft((arrS*arrAera).reshape(-1), arrK_Sp.reshape(-1,2), arrK_Ct.reshape(-1,2)).reshape(nPix,nPix)

# plot
figure()
subplot(121)
imshow(abs(arrM0), cmap='gray')
title("Original Image")
subplot(122)
imshow(abs(imgReco), cmap='gray')
title("NUDFT Reconstruction")

show()
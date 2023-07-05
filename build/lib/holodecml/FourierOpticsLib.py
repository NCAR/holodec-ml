"""
Routines to handle phase shift and power normalization
with numpy fft routines
"""

import numpy as np

def OpticsFFT(Ain):
    """
    Apply 2D fft to input matrix
    """
    Aout = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ain)))/np.sqrt(np.size(Ain))
    return Aout

def OpticsIFFT(Ain):
    """
    Apply 2D inverse fft to input matrix
    """
    Aout = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Ain)))*np.sqrt(np.size(Ain))
    return Aout
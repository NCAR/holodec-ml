"""
Routines for leveraging Pytorch in Optical
Calculations
"""

import torch
import numpy as np

def torch_holo_set(Ein:torch.tensor,
        fx:torch.tensor,
        fy:torch.tensor,
        z_tnsr:torch.tensor,
        lam:float)->torch.tensor:
    """
    Propagates an electric field a distance z

    Ein complex torch.tensor
    - input electric field
    
    fx:real torch.tensor
    - x frequency axis (3D, setup to broadcast)
    
    fy: real torch.tensor
    - y frequency axis (3D, setup to broadcast)
    
    z_tnsr: torch.tensor
    - tensor of distances to propagate the wave Ein
        expected to have dims (Nz,1,1) where Nz is the number of z
        dimensions
    
    lam: float
    - wavelength
    
    returns: complex torch.tensor with dims (Nz,fy,fx)
    
    Note the torch.fft library uses dtype=torch.complex64
    This may be an issue for GPU implementation
    
    """
    Etfft = torch.fft.fft2(Ein)
    # mult = 2*np.pi/lam*torch.sqrt(1-lam**2*(fx**2+fy**2))
    # Eofft = Etfft*torch.exp(1j*mult*z_tnsr)
    Eofft = Etfft*torch.exp(1j*2*np.pi*z_tnsr/lam*torch.sqrt(1-lam**2*(fx**2+fy**2)))
    
    # It might be helpful if we could omit this step.  It would save an inverse fft.
    Eout = torch.fft.ifft2(Eofft)
    
    return Eout
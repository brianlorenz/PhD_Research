import numpy as np
import matplotlib.pyplot as plt

def gaussian_lsf_kernel(wavelength, R, size=100):
    # Calculate FWHM and sigma
    FWHM = wavelength / R
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    # Define range and kernel
    x = np.linspace(-3*FWHM, 3*FWHM, size)
    kernel = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)
    
    # Normalize kernel
    kernel /= np.sum(kernel)
    
    return x, kernel

# Example usage:
wavelength = 13000  # in Angstroms
R = 35
x, kernel = gaussian_lsf_kernel(wavelength, R)

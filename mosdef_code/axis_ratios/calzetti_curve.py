import numpy as np
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd 

# Define the wavelength range in angstroms
wavelength = np.linspace(2000, 8000, 1000) # angstroms

# Convert wavelength to microns for use in the Calzetti curve
wavelength_microns = wavelength / 10000.0

# Define the Calzetti dust attenuation curve
def calzetti_curve(wavelength):
    k_lambda = np.zeros_like(wavelength)
    idx = np.where((wavelength >= 6300) & (wavelength <= 22000))[0]
    k_lambda[idx] = 2.659 * (-1.857 + 1.040 / wavelength_microns[idx]) + 4.05
    idx = np.where((wavelength >= 1200) & (wavelength < 6300))[0]
    k_lambda[idx] = 2.659 * (-2.156 + 1.509 / wavelength_microns[idx] - 0.198 / wavelength_microns[idx]**2 + 0.011 / wavelength_microns[idx]**3) + 4.05
    return k_lambda

def calzetti_law(wavelength_um):
    if wavelength_um >= 0.6300 and wavelength_um <= 2.2000:
        k_lambda = 2.659 * (-1.857 + 1.040 / wavelength_um) + 4.05
    if wavelength_um >= 0.1200  and wavelength_um < 0.6300:
        k_lambda = 2.659 * (-2.156 + 1.509 / wavelength_um - 0.198 / wavelength_um**2 + 0.011 / wavelength_um**3) + 4.05
    return k_lambda

breakpoint()
# Plot the Calzetti dust attenuation curve
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(wavelength, calzetti_curve(wavelength), color='red')
ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
ax.set_ylabel('Attenuation', fontsize=14)
ax.set_yticks([])
ax.tick_params(labelsize=14)

fig.savefig(imd.mosdef_dir + '/talk_plots/calzetti.pdf', bbox_inches='tight')
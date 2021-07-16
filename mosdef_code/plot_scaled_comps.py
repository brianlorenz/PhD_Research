import os
from astropy.io import ascii
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt




def plot_scaled_composites(n_clusters):
    """Using the scaling that was done above, plot the scaled composite seds
    
    Parameters:
    n_clusters (int): Number of clusters
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for groupID in range(n_clusters):
        data_df = ascii.read(imd.composite_sed_csvs_dir + f'/{groupID}_sed.csv').to_pandas()
        if len(os.listdir(imd.cluster_dir + f'/{groupID}')) < 15:
            continue
        ax.plot(data_df['rest_wavelength'], data_df['rest_wavelength']*data_df['f_lambda_scaled'], ls='-', marker='None')
        # ax.plot([5000, 5000], [-10, 10], ls='--', color='black')
    ax.set_xscale('log')
    ax.set_xlim(800, 45000)
    # ax.set_ylim(-1e-16, 9e-16)
    ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Normalized Flux ($\lambda$ F$_\lambda$)', fontsize=14)
    ax.tick_params(size=12)
    fig.savefig(imd.composite_sed_images_dir + '/scaled_composites.pdf')


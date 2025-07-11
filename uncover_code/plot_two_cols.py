import matplotlib.pyplot as plt
from astropy.io import ascii
from plot_vals import *
from full_phot_read_data import read_merged_lineflux_cat
# from full_phot_plots import make_cuts_lineflux_df


def plot_two_cols(df, colX_name, colY_name):
    fig, ax=plt.subplots(figsize=(7,6))

    ax.plot(df[colX_name], df[colY_name], ls='None', marker='o', color='black')

    ax.set_xlabel(colX_name, fontsize=14)
    ax.set_ylabel(colY_name, fontsize=14)
    ax.tick_params(labelsize=14)
    scale_aspect(ax)

    plt.show()

if __name__ == '__main__':
    # ha_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_Halpha.csv').to_pandas()
    # colX_name = 'Halpha_snr'
    # colY_name = 'Halpha_quality_factor'

    # plot_two_cols(ha_df, colX_name, colY_name)

    # merged_lineflux_df = read_merged_lineflux_cat()
    # plot_two_cols(merged_lineflux_df, 'Halpha_quality_factor', 'Halpha_chi2_scaled')
    breakpoint()
    
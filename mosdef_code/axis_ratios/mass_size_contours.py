import numpy as np
import matplotlib.pyplot as plt
from axis_ratio_funcs import read_filtered_ar_df
from plot_vals import *
import seaborn as sns
import initialize_mosdef_dirs as imd

def plot_mass_size_contours():
    ar_df = read_filtered_ar_df()
    mass_points = ar_df['log_mass']
    size_points = ar_df['half_light']

    fig, ax = plt.subplots(figsize = (8,8))

    low_axis = ar_df['use_ratio'] < 0.55
    high_axis = ~low_axis

    
    sns.kdeplot(mass_points[low_axis], size_points[low_axis], levels=[0.68, 0.95], ax=ax, color=light_color)
    # sns.kdeplot(mass_points[low_axis], size_points[low_axis], levels=[0.95], ax=ax, color=light_color)
    sns.kdeplot(mass_points[high_axis], size_points[high_axis], levels=[0.68, 0.95], ax=ax, color=dark_color)
    # sns.kdeplot(mass_points[high_axis], size_points[high_axis], levels=[0.95], ax=ax, color=dark_color)
    ax.set_xlabel(stellar_mass_label, fontsize = single_column_axisfont)
    ax.set_ylabel('R$_e$', fontsize = single_column_axisfont)
    ax.tick_params(labelsize = single_column_axisfont)
    fig.savefig(imd.axis_output_dir + '/mass_size_contour_nopoints.pdf')

    ax.plot(mass_points[low_axis], size_points[low_axis], color=light_color, marker='o', ls='None', label='Axis Ratio < 0.55')
    ax.plot(mass_points[high_axis], size_points[high_axis], color=dark_color, marker='o', ls='None', label='Axis Ratio >= 0.55')
    ax.legend(fontsize = single_column_axisfont-4)
    fig.savefig(imd.axis_output_dir + '/mass_size_contour.pdf')

plot_mass_size_contours()
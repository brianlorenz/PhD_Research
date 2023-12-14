import numpy as np
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii

stellar_mass_label = 'log$_{10}$(M$_*$ / M$_\odot$)'
av_extra_label = 'A$_V$ HII - A$_V$ star'
sfr_label = 'log$_{10}$(SFR) (M$_\odot$ / yr)'
ssfr_label = 'log$_{10}$(sSFR) (yr$^{-1}$)'
balmer_label = 'Balmer decrement (H$_\\alpha$ / H$_\\beta$)'
a_balmer_label = 'A$_\mathrm{Balmer}$'
balmer_av_label = 'Nebular A$_V$'
metallicity_label = '12 + log(O/H)'

single_column_axisfont = 24
single_column_ticksize = 24

full_page_axisfont = 18

light_color = '#DF7B15'
dark_color = '#2D1B99'

grey_point_color = '#BBBBBB'
grey_point_size = 3

cluster_marker = 's'
cluster_marker_color = 'blue'
cluster_marker_size = 8

paper_marker_edge_width = 1
paper_mec = 'black'
paper_marker_size = 10



# Turns the plot into a square, mainitaining the axis limits you set
def scale_aspect(ax):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)

def set_aspect_1(ax):
    """Forces the axis into a square"""
    ax.set_aspect(1./ax.get_data_ratio())

number_color = 'darkgrey'


def get_row_color(groupID):
    color_df = pd.read_csv(imd.loc_color_df)
    color_row = color_df[color_df['groupID']==groupID]
    rgba = color_row['rgba'].iloc[0]
    rgba = rgba.replace('(','')
    rgba = rgba.replace(')','')
    rgba = rgba.replace(',','')
    rgba = rgba.split(' ')
    rgba = [float(value) for value in rgba]
    return rgba

def get_row_size(groupID):
    group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv')
    size = 2.2*np.sqrt(len(group_df)) 
    return size 

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
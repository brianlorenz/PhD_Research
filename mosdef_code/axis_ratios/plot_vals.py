import numpy as np

stellar_mass_label = 'log$_{10}$(M$_*$ / M$_\odot$)'
sfr_label = 'log$_{10}$(SFR) (M$_\odot$ / yr)'
ssfr_label = 'log$_{10}$(sSFR) (yr$^{-1}$)'
balmer_label = 'Balmer decrement (H$_\\alpha$ / H$_\\beta$)'
balmer_av_label = 'A$_\mathrm{Balmer}$'
metallicity_label = '12 + log(O/H)'

single_column_axisfont = 24
single_column_ticksize = 24

full_page_axisfont = 20

light_color = '#DF7B15'
dark_color = '#2D1B99'


# Turns the plot into a square, mainitaining the axis limits you set
def scale_aspect(ax):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
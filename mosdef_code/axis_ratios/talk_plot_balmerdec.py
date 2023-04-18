from stack_fake_em_lines import generate_fake_galaxy_prop
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
import initialize_mosdef_dirs as imd

def generate_balmer_demo_separateplots():
    """Old, do not use - just here if wanting to go back and separate the plots
    """
    line_peaks = [4861, 6563]
    nodust_gal_df = generate_fake_galaxy_prop(0, 1, 100, line_peaks, 2.86, wavelength_res=0.1)
    dusty_gal_df = generate_fake_galaxy_prop(0, 0.9, 100, line_peaks, 6.0, wavelength_res=0.1)

    fig = plt.figure(figsize=(6, 8))
    
    n_rows = 2
    axarr = GridSpec(3, 1, left=0.08, right=0.92, wspace=0.13, hspace=0.15,height_ratios=[1,0.3,1])
    plot_lims = ((4853, 4870), (6552, 6574))
    bax_top = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
    bax_bot = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,0])

    def plot_on_axis(df, bax, color, text):
        bax.plot(df['rest_wavelength'], df['f_lambda_norm'], color=color)
        bax.set_xlabel('Wavelength ($\AA$)', fontsize = 14, labelpad=25)
        bax.set_ylabel('Flux', fontsize = 14, labelpad=45)
        bax.tick_params(labelsize = 14)
        bax.text(4856, 0.16, text, fontsize=14, color=color)
    
    bax_top.set_ylim(0, 0.19)
    bax_bot.set_ylim(0, 0.19)
    plot_on_axis(nodust_gal_df, bax_top, 'blue', 'No Dust')
    plot_on_axis(dusty_gal_df, bax_bot, 'orange', 'Dusty')

    fig.savefig(imd.mosdef_dir+'/talk_plots/talk_plot_balmerdec.pdf', bbox_inches='tight')


def generate_balmer_demo_overlaid():
    line_peaks = [4861, 6563]
    nodust_gal_df = generate_fake_galaxy_prop(0, 1, 100, line_peaks, 2.86, wavelength_res=0.1)
    dusty_gal_df = generate_fake_galaxy_prop(0, 1, 100, line_peaks, 6.0, wavelength_res=0.1)

    fig = plt.figure(figsize=(6, 4))
    
    n_rows = 2
    axarr = GridSpec(1, 1, left=0.08, right=0.92, wspace=0.13, hspace=0.15)
    plot_lims = ((4853, 4870), (6552, 6574))
    bax = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])

    def plot_on_axis(df, bax, color, label, ls='-'):
        bax.plot(df['rest_wavelength'], df['f_lambda_norm'], color=color, label=label, ls=ls)
        bax.set_xlabel('Wavelength ($\AA$)', fontsize = 14, labelpad=25)
        bax.set_ylabel('Normalized Flux', fontsize = 14, labelpad=45)
        bax.tick_params(labelsize = 14)
        # bax.text(4856, 0.16, text, fontsize=14, color=color)

    plot_on_axis(nodust_gal_df, bax, 'blue', 'No Dust')
    plot_on_axis(dusty_gal_df, bax, 'orange', 'Dusty', ls='--')
    bax.legend(fontsize=14, loc=2)

    fig.savefig(imd.mosdef_dir+'/talk_plots/talk_plot_balmerdec_overlaid.pdf', bbox_inches='tight')

generate_balmer_demo_overlaid()

# Copies all the figures from the paper into the paper figures folder:
import initialize_mosdef_dirs as imd
import os
import time
import stat
import shutil


# Save name of the directory to pull from
# save_name = 'both_sfms_4bin_median_2axis_boot100'
save_name = 'norm_1_sn5_filtered'
destination = imd.axis_output_dir + '/paper_figures'

# List of the paths to all the figures
fig_list = [
    imd.axis_cluster_data_dir + f'/{save_name}/sample_cut_2panel.pdf',
    imd.axis_output_dir + f'/axis_ratio_compare.pdf',
    imd.axis_output_dir + f'/axis_ratio_histogram.pdf',
    imd.axis_output_dir + f'/Dust_Diagram1_clumps.pdf',
    imd.axis_output_dir + f'/Dust_Diagram2_clumps.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/av_extra_mass.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_sfr_metallicity.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_mass.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/av_beta_combined.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/mass_metallicity.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/metallicity_sfr_sanders.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/uvj_ar_groups.pdf',
    imd.axis_cluster_data_dir + f'/{save_name}/overlaid_spectra_paper_8panel.pdf'
    
]



def copy_figure(fig_path):
    fig_name = fig_path.split('/')[-1]


    fileStatsObj = os.stat(fig_path)
    modificationTime = time.ctime(fileStatsObj[stat.ST_MTIME])
    print(f"Last Modified {fig_name}: ", modificationTime)

    target = destination + '/' + fig_name
    shutil.copyfile(fig_path, target)



def copy_all_figures(fig_list):
    for fig_path in fig_list:
        copy_figure(fig_path)

copy_all_figures(fig_list)
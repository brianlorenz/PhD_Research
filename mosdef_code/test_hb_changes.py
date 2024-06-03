import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

mosdef_new = ascii.read(imd.mosdef_dir +'/axis_ratio_data/Merged_catalogs/mosdef_all_cats_v2.csv').to_pandas()
mosdef_old = ascii.read(imd.mosdef_dir +'/axis_ratio_data/Merged_catalogs/mosdef_all_cats_v2_20240522.csv').to_pandas()

def compute_metals(group_df, snr_thresh=3):
    group_df['hb_SNR'] = group_df['hb_flux'] / group_df['err_hb_flux']
    group_df['nii_6585_SNR'] = group_df['nii_6585_flux'] / group_df['err_nii_6585_flux']
    hb_detected_rows = group_df['hb_SNR']>snr_thresh
    nii_detected_rows = group_df['nii_6585_SNR']>snr_thresh
    both_detected = np.logical_and(nii_detected_rows, hb_detected_rows)
    group_df['log_recomputed_ssfr'] = np.log10(group_df['recomputed_sfr']/(10**group_df['log_mass']))
    #Compute metallicity
    group_df['N2Ha'] = group_df['nii_6585_flux'] / group_df['ha_flux']
    group_df['O3Hb'] = group_df['oiii_5008_flux'] / group_df['hb_flux']
    group_df['log_O3N2'] = np.log10(group_df['O3Hb'] / group_df['N2Ha']) 
    group_df['O3N2_metallicity'] = 8.97-0.39*np.log10(group_df['O3Hb'] / group_df['N2Ha'])
    return group_df


fig, axarr = plt.subplots(2, 3, figsize=(16,6))
ha_ax = axarr[0,0]
hb_ax = axarr[0,1]
met_ax = axarr[0,2]
ha_mosdef = axarr[1,0]
hb_mosdef = axarr[1,1]
ax_list = [ha_ax, hb_ax, met_ax, ha_mosdef, hb_mosdef]
for groupID in range(20):
    group_df_new = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
    group_df_original = ascii.read(imd.mosdef_dir + f'/Clustering/cluster_indiv_dfs_original/{groupID}_cluster_df.csv').to_pandas()
    group_df_new = compute_metals(group_df_new)
    group_df_original = compute_metals(group_df_original)
    hb_good_idxs = np.logical_and(group_df_original['hb_flux']>-90, group_df_new['hb_flux']>-90)
    ha_good_idxs = np.logical_and(group_df_original['ha_flux']>-90, group_df_new['ha_flux']>-90)
    met_good_idxs = np.logical_and(ha_good_idxs, group_df_original['O3N2_metallicity']>6)
    met_good_idxs = np.logical_and(met_good_idxs, group_df_original['O3N2_metallicity']<10)
    
    hb_ax.plot(group_df_original[hb_good_idxs]['hb_flux'], group_df_new[hb_good_idxs]['hb_flux'], marker='o', ls='None', color='black', ms=3)
    ha_ax.plot(group_df_original[ha_good_idxs]['ha_flux'], group_df_new[ha_good_idxs]['ha_flux'], marker='o', ls='None', color='black', ms=3)
    met_ax.plot(group_df_original[met_good_idxs]['O3N2_metallicity'], group_df_new[met_good_idxs]['O3N2_metallicity'], marker='o', ls='None', color='black', ms=3)
mosdef_hahb_good = np.logical_and(mosdef_new['ha_flux']>-90, mosdef_new['hb_flux']>-90)
hb_mosdef.plot(mosdef_old[mosdef_hahb_good]['hb_flux'], mosdef_new[mosdef_hahb_good]['hb_flux'], marker='o', ls='None', color='black', ms=3)
ha_mosdef.plot(mosdef_old[mosdef_hahb_good]['ha_flux'], mosdef_new[mosdef_hahb_good]['ha_flux'], marker='o', ls='None', color='black', ms=3)
for ax in ax_list:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, ls='--', color='red', zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
ha_ax.set_title('halpha', fontsize=16)
ha_ax.set_xlabel('halpha original')
ha_ax.set_ylabel('halpha corrected')
hb_ax.set_title('hbeta', fontsize=16)
hb_ax.set_xlabel('hbeta original')
hb_ax.set_ylabel('hbeta corrected')
met_ax.set_title('metallicity', fontsize=16)
ha_mosdef.set_title('halpha', fontsize=16)
ha_mosdef.set_xlabel('halpha original mosdefdf')
ha_mosdef.set_ylabel('halpha new mosdefdf')
hb_mosdef.set_title('hblpha', fontsize=16)
hb_mosdef.set_xlabel('hbeta original mosdefdf')
hb_mosdef.set_ylabel('hbeta new mosdefdf')
met_ax.set_xlabel('metallicity original')
met_ax.set_ylabel('metallicity corrected')
plt.show()



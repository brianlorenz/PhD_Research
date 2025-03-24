# Lots of code from sps_quicklook_sed_sfh_posterior_v3.ipynb

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from astropy.table import Table
import pandas as pd
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
from astropy.io import ascii
from uncover_make_sed import get_sed

def get_prospect_df_loc(id_msa, id_dr3=False):
    loc_prospector_spec_df = f'/Users/brianlorenz/uncover/Data/prospector_spec/{id_msa}_prospector_spec_df.csv'
    loc_prospector_sed_df = f'/Users/brianlorenz/uncover/Data/prospector_spec/{id_msa}_prospector_sed_df.csv'
    if id_dr3:
        loc_prospector_spec_df = f'/Users/brianlorenz/uncover/Data/prospector_spec_dr3/{id_msa}_prospector_spec_df.csv'
        loc_prospector_sed_df = f'/Users/brianlorenz/uncover/Data/prospector_spec_dr3/{id_msa}_prospector_sed_df.csv'
    return loc_prospector_spec_df, loc_prospector_sed_df

def make_prospector(id_msa, plt_jy=True, id_dr3=False):
    objid = id_msa
    loc_prospector_spec_df, loc_prospector_sed_df = get_prospect_df_loc(id_msa, id_dr3=id_dr3)

    folder = '/Users/brianlorenz/uncover/Data/latest_zspec/'

    fver = 'zspec_dr4_v5.2.0_LW_SUPER_spsv0.2' # DR 4 zspec

    ffsed = folder + 'ancillaries/seds_map_{}.npz'.format(fver)
    ffsfh = folder + 'ancillaries/sfhs_{}.npz'.format(fver)
    ffchain = folder + 'ancillaries/chains_{}.npz'.format(fver)

    # load data
    fsed = np.load(ffsed, allow_pickle=True)
    fsfh = np.load(ffsfh, allow_pickle=True)
    fchain = np.load(ffchain, allow_pickle=True)

    if objid not in fsed['objid']:
        print('no sed for obj {}!')
        return None
        
    _idx = np.squeeze(np.where(fsed['objid']==objid))

    zred = fsed['zred'][_idx]
    print('zml:', zred)
    mu = fsed['mu'][_idx]
    obsmags = fsed['obsmags'][_idx]
    obsunc = fsed['obsunc'][_idx]
    wavspec = fsed['wavspec'] * (1+zred)
    
    weff = fsed['weff'] / 1e4
    wavspec = wavspec / 1e4
    
    modmags = fsed['modmags'][_idx] * mu
    modspec = fsed['modspec'][_idx] * mu
    
    mask = np.isfinite(obsmags)
    obsmags = obsmags[mask]
    obsunc = obsunc[mask]
    weff = weff[mask]
    modmags = modmags[mask]
    obsunc = np.clip(obsunc, a_min=obsmags*0.05, a_max=None)
        
    if plt_jy:
        obsmags *= 3631
        obsunc *= 3631
        modmags *= 3631
        modspec *= 3631

    # phot.errorbar(weff, modmags, fmt='o', color='firebrick', label='model photometry', zorder=100,
    #             elinewidth=1, mec='k', mew=0.2)

    # phot.errorbar(weff, obsmags, yerr=obsunc, color='black', fmt='o', label='observed photometry', zorder=101)
    # phot.plot(wavspec, modspec, '-', color='firebrick', label = 'model spectrum', zorder=-100)
    prospector_spec_df = pd.DataFrame(zip(wavspec, modspec), columns = ['wave_um', 'flux_jy'])
    prospector_spec_df['wave_aa'] = 10000 * prospector_spec_df['wave_um']
    prospector_spec_df.to_csv(loc_prospector_spec_df, index=False)

    prospector_sed_df = pd.DataFrame(zip(weff, modmags), columns = ['weff_um', 'flux_jy'])
    prospector_sed_df['weff_aa'] = 10000 * prospector_sed_df['weff_um']
    prospector_sed_df.to_csv(loc_prospector_sed_df, index=False)
    
    return prospector_spec_df

def read_prospector(id_msa):
    loc_prospector_spec_df, loc_prospector_sed_df = get_prospect_df_loc(id_msa)
    prospector_spec_df = ascii.read(loc_prospector_spec_df).to_pandas()
    prospector_sed_df = ascii.read(loc_prospector_sed_df).to_pandas()
    return prospector_spec_df, prospector_sed_df


def plot_prospector_vs_sed(id_msa):
    prospector_spec_df, prospector_sed_df = read_prospector(id_msa)
    sed_df = get_sed(id_msa)
    # breakpoint()
    fig, ax = plt.subplots(figsize=(6,6))
    wave_range_spec = np.logical_and(prospector_spec_df['wave_um'] > 0.3, prospector_spec_df['wave_um'] < 6)
    wave_range_sed = np.logical_and(prospector_sed_df['weff_um'] > 0.3, prospector_sed_df['weff_um'] < 6)

    ax.plot(prospector_spec_df[wave_range_spec]['wave_um'], prospector_spec_df[wave_range_spec]['flux_jy'], color = 'black', ls='-', marker='None', label='Prospector Spec')
    ax.plot(prospector_sed_df[wave_range_sed]['weff_um'], prospector_sed_df[wave_range_sed]['flux_jy'], color = 'purple', ls='None', marker='o', label='Prospector SED')
    ax.plot(sed_df['eff_wavelength'], sed_df['flux'], color = 'orange', ls='None', marker='o', label = 'Observed SED')
    ax.legend()
    plt.show()

def make_all_prospector(id_msa_list):
    for id_msa in id_msa_list:
        make_prospector(id_msa, plt_jy=True)

if __name__ == "__main__":
    id_msa = 47875
    # zqual_df = read_spec_cat()
    # redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]
    # prospector_spec_df, prospector_sed_df = read_prospector(id_msa)
    # pab_region = np.logical_and(prospector_spec_df['wave_um'] > 1.0*(1+redshift), prospector_spec_df['wave_um']<1.5*(1+redshift))
    # plt.plot(prospector_spec_df[pab_region]['wave_um']/(1+redshift), prospector_spec_df[pab_region]['flux_jy']*(1+redshift))
    # breakpoint()
    # plot_prospector_vs_sed(id_msa)
    make_prospector(id_msa, plt_jy=True)

            
    # zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    # id_msa_list = zqual_detected_df['id_msa'].to_list()
    # make_all_prospector(id_msa_list)
    pass

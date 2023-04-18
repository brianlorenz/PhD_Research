import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary, describe
from prospect.sources import CSPSpecBasis, FastStepBasis

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from sedpy import observate
from astropy.io import fits
from scipy import signal
import dynesty
import h5py
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import truncnorm
import os
import sys
import time
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
import glob
from astropy.cosmology import z_at_value
from astropy.io import ascii
from astropy.io import fits


# Directory locations on savio
# composite_sed_csvs_dir = '/global/scratch/users/brianlorenz/composite_sed_csvs'
# composite_filter_sedpy_dir = '/global/scratch/users/brianlorenz/sedpy_par_files'
# median_zs_file = '/global/scratch/users/brianlorenz/median_zs.csv'

# Directory locations on home
import initialize_mosdef_dirs as imd
composite_sed_csvs_dir = imd.composite_sed_csvs_dir
composite_filter_sedpy_dir = imd.composite_filter_sedpy_dir
median_zs_file = imd.composite_seds_dir + '/median_zs.csv'

# %run prospector_dynesty.py --param_file='prospector_composite_params_nonpar_localtest.py' --outfile='composite_group0' --debug=True
# "%run prospector_dynesty.py --param_file='prospector_composite_params_group4_trial0.py' --outfile='/global/scratch/users/brianlorenz/prospector_h5s/nonpar_sfh_fixedbins_h5s/group2_trial19'  --debug=True"

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

# --------------
# Run Params Dictionary Setup
# --------------


run_params = {'verbose': True,
              'debug': False,
              'outfile': 'composite',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # --- dynesty parameters ---
              'nested_bound': 'multi',        # bounding method
              'nested_sample': 'rwalk',       # sampling method
              'nested_walks': 70,     # sampling gets very inefficient w/ high S/N spectra
              'nested_nlive_init': 600, # a finer resolution in likelihood space
              'nested_nlive_batch': 100,
              'nested_maxbatch': None, # was None-- changed re ben's email 5/21/19
              'nested_maxcall': float(1e7), # was 5e7 -- changed to 5e6 re ben's email on 5/21/19
              'nested_bootstrap': 20,
              'nested_dlogz_init': 0.05,
              'nested_stop_kwargs': {"post_thresh": 0.5}, # lower this to 0.02ish once I get things working
              # Obs data parameters
              'objid': 0,
              'zred': 0.0,
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              'groupID': 1,
              }

# --------------
# OBS
# --------------


def build_obs(**kwargs):
    """Load the obs dict

    :returns obs:
        Dictionary of observational data.
    """
    groupID = run_params['groupID']
    sed_file = composite_sed_csvs_dir + f'/{groupID}_sed.csv'
    filt_folder = composite_filter_sedpy_dir + f'/{groupID}_sedpy_pars'

    # test
    print(f'Loading object {groupID}')

    # set up obs dict
    obs = {}

    zs_df = ascii.read(median_zs_file).to_pandas()
    obs['z'] = zs_df[zs_df['groupID'] == groupID]['median_z'].iloc[0]

    # load photometric filters
    obs["filters"] = get_filt_list(filt_folder)

    # load photometry
    sed_data = ascii.read(sed_file).to_pandas()
    obs["phot_wave"] = sed_data['redshifted_wavelength'].to_numpy()
    obs['maggies'] = (sed_data['f_maggies_red']).to_numpy()
    #obs['maggies_unc'] = (sed_data['err_f_maggies_avg']).to_numpy()
    # Add 5 percent in quadrature to errors
    data_05p = (sed_data['f_maggies_red']) * 0.05
    obs['maggies_unc'] = (np.sqrt(data_05p**2 + (sed_data['err_f_maggies_avg_red'])**2)).to_numpy()
    
    # Phot mask that allows everything
    # obs["phot_mask"] = np.array([m > 0 for m in obs['maggies']])
    # Phot mask around emission lines
    obs["phot_mask"] = check_filt_transmission(filt_folder, obs['z'])
    # Phot mask out anything blueward of 1500
    redshifted_lya_cutoff = 1500*(1+obs['z'])
    ly_mask = obs["phot_wave"] > redshifted_lya_cutoff
    breakpoint()

    # Add unessential bonus info.  This will be stored in output
    obs['groupID'] = groupID
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['mask'] = None
    obs['unc'] = None

    return obs


# This is the default from prospector
def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=True,
                add_neb=True, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    groupID = run_params['groupID']

    model_params = TemplateLibrary["continuity_sfh"]
    
    #   This accounts for the resolution of the spectrum: convolve model to match data resolution, 
    #   fit for velocity dispersion
    #model_params.update(TemplateLibrary["spectral_smoothing"])
    
    #   test this: optimize out moderate-order polynomial at every likelihood call
    #model_params.update(TemplateLibrary['optimize_speccal'])

    # set IMF to chabrier (default is kroupa)
    # model_params['imf_type']['init'] = 1

    zs_df = ascii.read(median_zs_file).to_pandas()
    median_z = zs_df[zs_df['groupID'] == groupID]['median_z'].iloc[0]
    model_params["zred"]['init'] = median_z

    ############## SFH #################
    ''' This fit: the "standard" fixed-bin setup that is preferred by Joel's paper.
         We will have a few old bins, and some shorter fixed bins. Our chosen model
        uses 9 bins, so replicate that here as well. Maybe 4 fixed old bins, then 5 young
          bins from 0-20, 20-50, 50-100, 100-200, 200-500 Myr'''
    # remember that agebins are in units of log(yrs) of lookback time
    # tuniv = cosmo.age(median_z).value
    # agelims_young = np.array([1, 20e6, 50e6, 100e6, 200e6, 500e6])
    # agelims_old = np.logspace(np.log10(500e6), np.log10(tuniv*1e9), 5)[1:]
    # agelims = np.concatenate((agelims_young, agelims_old))
    # agebins = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T     
    # ncomp = agebins.shape[0]     
    
    # get umachine priors on logsfr_ratios and logmass - need umachine files
    # logsfr_ratios_init, logmass_init = umachine_priors(agebins, median_z)
            
    # load nvariables and agebins
    # model_params['mass'] = {'N': ncomp, 'init': np.full(ncomp,1e6), 'depends_on': massmet_to_masses,
    #     'isfree':False}
    # Testing without umachine priors
    # model_params['agebins']['N'] = ncomp
    # model_params['agebins']['init'] = agebins
    # model_params['agebins']['isfree'] = False
    # model_params['logsfr_ratios']['N'] = ncomp - 1
    # model_params['logsfr_ratios']['init'] = logsfr_ratios_init
    # model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=logsfr_ratios_init,
    #     scale=np.ones(ncomp-1) * 0.3, df=np.ones(ncomp-1)*2) 

    # MOre parameters and mass-met dependence in wren's email 10/14/22

    # Can modfiy default priors
    # model_params["logzsol"]["prior"] = priors.TopHat(mini=-2, maxi=0.19)
    # model_params["dust2"]["prior"] = priors.TopHat(mini=0, maxi=2)
    # model_params["logmass"]["prior"] = priors.TopHat(mini=7, maxi=12)

    # Can edit number of bins or priors on SFH - don't fully udnerstand this
    # nbins_sfh = 8
    # model_params['agebins']['N'] = nbins_sfh
    # model_params['mass']['N'] = nbins_sfh
    # model_params['logsfr_ratios']['N'] = nbins_sfh-1
    # model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    # model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
    #                                                               scale=np.full(nbins_sfh-1,0.3),
    #                                                               df=np.full(nbins_sfh-1,2))
                       
    # Can add dust parameters
    model_params.update(TemplateLibrary["nebular"])
    # model_params.update(TemplateLibrary["dust_emission"])

    # Now instantiate the model using this new dictionary of parameter
    # specifications
    model = sedmodel.SedModel(model_params)

    return model


# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    # Parametric
    # sps = CSPSpecBasis(zcontinuous=zcontinuous,
    #                    compute_vega_mags=compute_vega_mags)
    # Non=parametric
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


# -----------------
# Helper Functions
# ------------------

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction * dust2


def to_median_redshift(wavelength, median_z):
    """Converts wavelength array to the median redshift of the sample

    Parameters:
    wavelength (array): wavelength values to convert
    median_z (int): redshift to change the wavelength by

    Returns:
    wavelength_red (array): redshifted wavelength

    """

    wavelength_red = wavelength * (1 + median_z)
    return wavelength_red


def get_filt_list(target_folder):
    """Uses sedpy to read in a list of filters when given a folder that contains them

    Parameters:
    target_folder (str) - location of folder containing sedpy filter .par files

    """
    filt_files = [file.replace('.par', '') for file in os.listdir(
        target_folder) if '_red.par' in file]
    filt_files.sort()
    filt_list = observate.load_filters(filt_files, directory=target_folder)
    return filt_list


def check_filt_transmission(target_folder, redshift, transmission_threshold = 0.70):
    """Makes a photometric mask from the filters by checking if each filter has a line with high transmission
    
    Parameters:
    target_folder: folder where the _zred.par files are stored
    redshift: z
    transmission_threshold: If line transmission is greater than this value, mask the pixel
    """
    emission_lines = [4863, 5008, 6565]
    filt_files = [file for file in os.listdir(target_folder) if '_red.par' in file]
    filt_files.sort()
    phot_mask = []
    for i in range(len(filt_files)):
        filt_df = ascii.read(target_folder + '/' + filt_files[i]).to_pandas()
        filt_df = filt_df.rename(columns={'col1':'obs_wave', 'col2':'transmission'})
        filt_df['rest_wave'] = filt_df['obs_wave']/(1+redshift)
        for line_center in emission_lines:
            abs = np.argmin(np.abs(filt_df['rest_wave']-line_center))
            line_transmission = filt_df.iloc[abs]['transmission']
            if line_transmission > transmission_threshold:
                # Mask the point and leave this loop
                mask_bool = False
                break
            else:
                # Don't mask 
                mask_bool = True
        phot_mask.append(mask_bool)
    phot_mask = np.array(phot_mask)
    return phot_mask

def umachine_priors(agebins=None, zred=None, **extras):
    '''set smarter priors on the logSFR ratios. given a 
    redshift zred, use the closest-z universe machine SFH
    to set a better guess than a constant SFR. returns
    agebins, logmass, and set of logSFR ratios that are
    self-consistent. '''
    
    tuniv = cosmo.age(zred).value
    
    # get the universe machine SFH at the closest redshift
    ufiles = glob.glob('umachine_SFH/*.dat')
    uma = np.array([float(a.split('_a')[1][:-4]) for a in ufiles])
    fname = ufiles[np.argmin(np.abs(uma-cosmo.scale_factor(zred)))]
    umachine = np.genfromtxt(fname, skip_header=10, names=True)
    a = float(fname.split('_')[-1][1:-4])
    # trim low-z stuff we don't need
    umachine = umachine[umachine['Avg_scale_factor'] <= a]
    zs = np.array([z_at_value(cosmo.scale_factor, uz) for uz in umachine['Avg_scale_factor']])
    ages = cosmo.age(zs).value

    # subsample to make sure we'll have umachine bin for everything
    factor = .001
    newages = np.arange(0, tuniv,factor)
    sfh = np.interp(newages, ages, umachine['SFH_Q'])
    # ok now make the time axis go from now backwards to match agebins
    sfh = sfh[::-1]
    mass = sfh*factor*1e9 # mass formed in each little bin    
    logmass = np.log10(np.sum(mass))  # total mass (return this so we can set prior here for consistency)
    
    # get default agebins in same units
    abins_age = 10**agebins/1e9 # in Gyr
    dt = (abins_age[:,1] - abins_age[:,0])*1e9 # in yr

    # get mass and SFR in each bin
    mUniv = np.zeros(len(abins_age))
    for i in range(0, len(abins_age)):
        mUniv[i] = np.sum(mass[(newages >= abins_age[i,0]) & (newages <= abins_age[i,1])])
    sfrUniv = mUniv / dt
    
    # then take the log ratio to get what we actually want back
    logsfr_ratios = np.log10(sfrUniv[:-1] / sfrUniv[1:])
    
    # return
    return(logsfr_ratios, logmass)

   # ipython -c "%run prospector_dynesty.py --param_file='prospector_composite_params_nonpar_localtest.py' --outfile='/Users/brianlorenz/mosdef/Clustering/local_test/nonpar_test'"
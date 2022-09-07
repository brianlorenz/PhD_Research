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
composite_sed_csvs_dir = '/global/scratch/users/brianlorenz/composite_sed_csvs'
composite_filter_sedpy_dir = '/global/scratch/users/brianlorenz/sedpy_par_files'
median_zs_file = '/global/scratch/users/brianlorenz/median_zs.csv'

# Directory locations on home
# import initialize_mosdef_dirs as imd
# composite_sed_csvs_dir = imd.composite_sed_csvs_dir
# composite_filter_sedpy_dir = imd.composite_filter_sedpy_dir
# median_zs_file = imd.composite_seds_dir + '/median_zs.csv'

# %run prospector_dynesty.py --param_file='prospector_composite_params_copy.py' --outfile='composite_group0' --debug=True

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
              # dynesty Fitter parameters
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'rwalk',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_stop_kwargs': {"post_thresh": 0.05},
              # Obs data parameters
              'objid': 0,
              'zred': 0.0,
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              'groupID': -1,
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

    model_params = TemplateLibrary["continuity_flex_sfh"]

    zs_df = ascii.read(median_zs_file).to_pandas()
    median_z = zs_df[zs_df['groupID'] == groupID]['median_z'].iloc[0]
    model_params["zred"]['init'] = median_z

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
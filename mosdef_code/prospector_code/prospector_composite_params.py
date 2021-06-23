import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis, FastStepBasis

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from sedpy.observate import load_filters
import sedpy
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
from convert_filter_to_sedpy import get_filt_list, find_median_redshift, to_median_redshift, de_median_redshift
from astropy.io import ascii
from astropy.io import fits


### SET THE GROUP HERE ###
composite_group = '3'


# %run prospector_dynesty.py --param_file='prospector_composite_params.py' --outfile='composite_group3_errs_inflate_dust1'

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

''' This fit: uses the flexible continuity SFH. Make sure to set last agebin to something relatively short: it and
the first agebin do not get to move, but the rest do. '''

# --------------
# Run Params Dictionary Setup
# --------------

''' Wren's params
run_params = {'verbose': True,  # this controls how much output to screen. If true, full chain gets spewed, so I normally turn off.
              'debug': False,  # if true, stops before mcmc burn in so you can look at model and stuff
              'outfile': 'squiggle',
              'output_pickles': False,
              'rescale_spectrum': False,  # we have real units on our spectrum
              # --- Optimization parameters ---
              # Do Levenburg optimization? (requires do_powell=False)
              'do_levenburg': False,
              # number of initial locations (drawn from prior) for
              # Levenburg-Marquardt
              'nmin': 10,
              # Do powell minimization? (deprecated)
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,  # Parameters for powell
              # --- emcee parameters ---
              'nwalkers': 128,                # number of emcee walkers
              # number of iterations in each burn-in round
              'nburn': [16, 16, 32, 64],
              'niter': 512,                   # number of iterations in production
              # fraction of production run at which to save to hdf5.
              'interval': 0.2,
              'initial_disp': 0.1,            # default dispersion in parameters for walker ball
              # Obs data parameters
              'objid': 0,
              'objname': 'spec-2438-54056-0396',
              'catfile': '../full_catalog.npy',
              'specfolder': '../all_psb_spectra/',
              'phottable': None,
              'logify_spectrum': False,
              'normalize_spectrum': False,
              # Model parameters
              'add_neb': True,
              'add_dust': True,
              # SPS parameters
              'zcontinuous': 1,
              'zred': 0.0,
              # --- SFH parameters ---
              'agelims': [0.0, 7.4772, 8.0, 8.5, 9.0, 9.5, 9.8, 10.0],
              'tquench': .2,
              'tflex': 2,
              'nflex': 5,
              'nfixed': 3,
              # --- dynesty parameters ---
              'nested_bound': 'multi',        # bounding method
              'nested_sample': 'rwalk',       # sampling method
              'nested_walks': 64,     # sampling gets very inefficient w/ high S/N spectra
              'nested_nlive_init': 600,  # a finer resolution in likelihood space
              'nested_nlive_batch': 100,
              'nested_maxbatch': None,  # was None-- changed re ben's email 5/21/19
              # was 5e7 -- changed to 5e6 re ben's email on 5/21/19
              'nested_maxcall': float(1e7),
              'nested_bootstrap': 20,
              'nested_dlogz_init': 0.05,
              # might want to lower this to 0.02ish once I get things working
              'nested_stop_kwargs': {"post_thresh": 0.2},
              # --- nestle parameters ---
              'nestle_npoints': 1000,         # Number of nested sampling live points
              'nestle_method': 'multi',       # Nestle sampling method
              # Maximum number of likelihood calls (ends even if not converged)
              'nestle_maxcall': int(1e7),
              }
              '''

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
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
              'objid': 0,
              'zred': 0.0,
              # Model parameters
              'add_neb': True,
              'add_duste': True,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------


def build_obs(objname=composite_group, **kwargs):
    """Load an SDSS spectrum.


    objname (str): Number of the composite SED group

             Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """

    sed_file = f'/Users/brianlorenz/mosdef/composite_sed_csvs/{objname}_sed.csv'
    filt_folder = f'/Users/brianlorenz/mosdef/composite_sed_csvs/composite_filter_csvs/{objname}_filter_csvs'

    # test
    print(f'Loading object {objname}')

    # set up obs dict
    obs = {}

    obs['z'] = find_median_redshift(objname)

    # load photometric filters
    obs["filters"] = get_filt_list(filt_folder)

    # load photometry
    sed_data = ascii.read(sed_file).to_pandas()
    obs["phot_wave"] = to_median_redshift(
        sed_data['rest_wavelength'], obs['z']).to_numpy()
    obs['maggies'] = (sed_data['f_maggies'] * 1000).to_numpy()
    #obs['maggies_unc'] = (sed_data['err_f_maggies_avg'] * 1000).to_numpy()
    # Add 5 percent in quadrature to errors
    data_05p = (sed_data['f_maggies'] * 1000) * 0.05
    obs['maggies_unc'] = (
        np.sqrt(data_05p**2 + (sed_data['err_f_maggies_avg'] * 1000)**2)).to_numpy()

    obs["phot_mask"] = np.array([m > 0 for m in obs['maggies']])
    # Mask out Halpha
    # obs["phot_mask"] = np.array(
    #     [np.logical_or(wave < 6000, wave > 7000) for wave in obs['phot_wave']])
    # Mask Ha and Hb
    # obs["phot_mask"] = np.array(
    #     [np.logical_and(np.logical_or(wave < 4500, wave > 5300), np.logical_or(wave < 6100, wave > 6900)) for wave in obs['phot_wave']])

    # print('OBS-- catalog: ' + row['specname'])

    ''' WHat is apcor?
    # apcor
    apcor = row['apcor']
    if apcor < 0:  # can be -99 sometimes
        apcor = np.median(cat['apcor'])


    # load in spectrum
    hdulist = fits.open(specfolder + objname + '.fits')
    obs['wavelength'] = 10**hdulist[1].data['loglam']  # in angstrom
    spec = 1e-17 * hdulist[1].data['flux'] * (10**hdulist[1].data['loglam'])**2. / 3e18 * \
        1e23 / 3631 * apcor  # in maggies!
    # signal.medfilt(spec, 9) # median filter the spectrum by 9
    obs['spectrum'] = spec
    obs['unc'] = 1e-17 / np.sqrt(hdulist[1].data['ivar']) * (10**hdulist[1].data['loglam'])**2. / 3e18 * \
        1e23 / 3631 * apcor  # in maggies!
    # make sure that everything has positive real uncertainties
    obs['mask'] = (np.isfinite(obs['unc'])) & (obs['unc'] > 0)
    '''

    '''update: mask forbidden lines for ALL spectra '''
    # now, check if this is an AGN. if so, mask a region around both [OIII] and [OII]
    # since AGN emission deeefinitely isn't in the models
    # if row['OIII_AGN'] == 1:
    # id regions to mask
    '''
    z = hdulist[2].data['z'][0]
    lines_to_mask = np.array([3727, 4958, 5007]) * (1 + z)  # oii, oiii doublet
    mask_width = np.array([50, 100, 100])
    # then set the mask flag (FALSE = masked, weirdly...)
    for i, l in enumerate(lines_to_mask):
        obs['mask'][np.abs(obs['wavelength'] - l) < mask_width[i]] = False

    print('OBS-- fits file: ' + specfolder + objname + '.fits')
    '''

    # Add unessential bonus info.  This will be stored in output
    obs['objname'] = objname
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['mask'] = None
    obs['unc'] = None
    # print('OBS-- redshift ' + str(obs['z']))
    # obs['apcor'] = row['apcor']  # add in apcor so it's there
    # hdulist.close()

    # print('OBS-- objname: ' + obs['objname'])

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

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]
    # model_params["dust1"] = {"name": "dust1", "N": 1, "isfree": True,
    #                          "init": 0.1, "units": "optical depth at 5500AA", "prior": priors.TopHat(mini=0.0, maxi=4.0)}

    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    # if luminosity_distance > 0:
    #     model_params["lumdist"] = {"N": 1, "isfree": False,
    #                                "init": luminosity_distance, "units": "Mpc"}

    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params['dust1'] = {'N': 1, 'isfree': False,
                             'depends_on': to_dust1, 'init': 1.0}
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0}
    model_params["logzsol"]["init"] = 0
    model_params["tage"]["init"] = 13.
    model_params["mass"]["init"] = 1e8

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    # model_params["mass"]["init_disp"] = 1e7
    # model_params["tau"]["init_disp"] = 3.0
    # model_params["tage"]["init_disp"] = 5.0
    # model_params["tage"]["disp_floor"] = 2.0
    # model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1_fraction"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e13)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        # And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    # Set to fit at median redshift
    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = find_median_redshift(composite_group)

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter
    # specifications
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
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

# if __name__ == '__main__':

#     # - Parser with default arguments -
#     parser = prospect_args.get_parser()
#     # - Add custom arguments -
#     parser.add_argument('--object_redshift', type=float, default=0.0,
#                         help=("Redshift for the model"))
#     parser.add_argument('--add_neb', action="store_true",
#                         help="If set, add nebular emission in the model (and mock).")
#     parser.add_argument('--add_duste', action="store_true",
#                         help="If set, add dust emission to the model.")
#     parser.add_argument('--luminosity_distance', type=float, default=1e-5,
#                         help=("Luminosity distance in Mpc. Defaults to 10pc "
#                               "(for case of absolute mags)"))
#     parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
#                         help="Names of table from which to get photometry.")
#     parser.add_argument('--objid', type=int, default=0,
#                         help="zero-index row number in the table to fit.")

#     args = parser.parse_args()
#     run_params = vars(args)
#     obs, model, sps, noise = build_all(**run_params)

#     run_params["sps_libraries"] = sps.ssp.libraries
#     run_params["param_file"] = __file__

#     print(model)

#     if args.debug:
#         sys.exit()

#     #hfile = setup_h5(model=model, obs=obs, **run_params)
#     hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
#     output = fit_model(obs, model, sps, noise, **run_params)

#     writer.write_hdf5(hfile, run_params, model, obs,
#                       output["sampling"][0], output["optimization"][0],
#                       tsample=output["sampling"][1],
#                       toptimize=output["optimization"][1],
#                       sps=sps)

#     try:
#         hfile.close()
#     except(AttributeError):
#         pass

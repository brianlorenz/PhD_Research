import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis, FastStepBasis
from sedpy.observate import load_filters
import sedpy
from astropy.io import fits
from scipy import signal
import dynesty
import h5py
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import truncnorm
import os
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
import glob
from astropy.cosmology import z_at_value

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

''' This fit: uses the flexible continuity SFH. Make sure to set last agebin to something relatively short: it and
the first agebin do not get to move, but the rest do. '''

# --------------
# Run Params Dictionary Setup
# --------------

run_params = {'verbose':True, #this controls how much output to screen. If true, full chain gets spewed, so I normally turn off.
              'debug':False, #if true, stops before mcmc burn in so you can look at model and stuff
              'outfile':'squiggle', 
              'output_pickles': False,
              'rescale_spectrum': False, # we have real units on our spectrum
              # --- Optimization parameters ---
              'do_levenburg': False,          # Do Levenburg optimization? (requires do_powell=False)
              'nmin': 10,                     # number of initial locations (drawn from prior) for Levenburg-Marquardt
              'do_powell': False,             # Do powell minimization? (deprecated)
              'ftol':0.5e-5, 'maxfev': 5000,  # Parameters for powell
              # --- emcee parameters ---
              'nwalkers': 128,                # number of emcee walkers
              'nburn': [16, 16, 32, 64],      # number of iterations in each burn-in round
              'niter': 512,                   # number of iterations in production
              'interval': 0.2,                # fraction of production run at which to save to hdf5.
              'initial_disp': 0.1,            # default dispersion in parameters for walker ball
              # Obs data parameters
              'objid':0,
              'objname':'spec-2438-54056-0396',
              'catfile': '../full_catalog.npy',
              'specfolder': '../all_psb_spectra/',
              'phottable':None,                                     
              'logify_spectrum':False,
              'normalize_spectrum':False,
              # Model parameters
              'add_neb': True,
              'add_dust': True,
              # SPS parameters
              'zcontinuous': 1,
              'zred': 0.0,
              # --- SFH parameters ---
              'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0], 
              'tquench': .2, 
              'tflex': 2, 
              'nflex': 5, 
              'nfixed': 3,
              # --- dynesty parameters ---
              'nested_bound': 'multi',        # bounding method
              'nested_sample': 'rwalk',       # sampling method
              'nested_walks': 64,     # sampling gets very inefficient w/ high S/N spectra
              'nested_nlive_init': 600, # a finer resolution in likelihood space
              'nested_nlive_batch': 100,
              'nested_maxbatch': None, # was None-- changed re ben's email 5/21/19
              'nested_maxcall': float(1e7), # was 5e7 -- changed to 5e6 re ben's email on 5/21/19
              'nested_bootstrap': 20,
              'nested_dlogz_init': 0.05,
              'nested_stop_kwargs': {"post_thresh": 0.02}, # might want to lower this to 0.02ish once I get things working
              # --- nestle parameters ---
              'nestle_npoints': 2000,         # Number of nested sampling live points
              'nestle_method': 'multi',       # Nestle sampling method
              'nestle_maxcall': int(1e7),     # Maximum number of likelihood calls (ends even if not converged)
              }

# --------------
# OBS
# --------------
def load_obs(objname='spec-2438-54056-0396', catfile=None, specfolder=None,
             luminosity_distance=None, **kwargs):
    """Load an SDSS spectrum.
             
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
             
             
    # test
    print(objname, catfile, specfolder)         
             
    # set up obs dict
    obs = {}

    # load photometric filters
    filternames = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']] \
                 + ['wise_w{}'.format(b) for b in ['1', '2', '3', '4']]
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # load photometry
    cat = np.load(catfile, allow_pickle=True)
    row = cat[cat['specname'] == objname+'.fits'][0] # catalog for this object
    sdss_phot = np.array([10**(-.4*row['mag_'+f]) for f in ['u','g','r','i','z']]) # in maggies
    sdss_phot_err = np.array([row['mag_'+f+'_err'] for f in ['u','g','r','i','z']]) * sdss_phot * (0.4*np.log(10))
    wise_phot = np.array([row['w'+str(f)+'_maggies'] for f in [1,2,3,4]]) # in maggies
    wise_phot_err = np.array([row['w'+str(f)+'_maggies_err'] for f in [1,2,3,4]])
    obs['maggies'] = np.concatenate([sdss_phot, wise_phot])
    obs['maggies_unc'] = np.concatenate([sdss_phot_err, wise_phot_err])
    # obs['maggies'] = sdss_phot
    # obs['maggies_unc'] = sdss_phot_err
    obs["phot_mask"] = np.array([m > 0 for m in obs['maggies']]) # what to include in the fit!
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]   
    
    print('OBS-- catalog: '+row['specname'])
    
    # apcor
    apcor = row['apcor']
    if apcor < 0: # can be -99 sometimes
        apcor = np.median(cat['apcor'])
    
    # load in spectrum
    hdulist = fits.open(specfolder+objname+'.fits')
    obs['wavelength'] = 10**hdulist[1].data['loglam'] # in angstrom
    spec = 1e-17*hdulist[1].data['flux'] * (10**hdulist[1].data['loglam'])**2. / 3e18 * \
            1e23 / 3631 * apcor # in maggies!
    obs['spectrum'] = spec #signal.medfilt(spec, 9) # median filter the spectrum by 9
    obs['unc'] = 1e-17/np.sqrt(hdulist[1].data['ivar']) * (10**hdulist[1].data['loglam'])**2. / 3e18 * \
            1e23 / 3631 * apcor # in maggies!
    obs['mask'] = (np.isfinite(obs['unc'])) & (obs['unc'] > 0) # make sure that everything has positive real uncertainties  
    
    '''update: mask forbidden lines for ALL spectra '''
    # now, check if this is an AGN. if so, mask a region around both [OIII] and [OII] 
    # since AGN emission deeefinitely isn't in the models
    # if row['OIII_AGN'] == 1:
    # id regions to mask
    z = hdulist[2].data['z'][0]
    lines_to_mask = np.array([3727, 4958, 5007]) * (1+z)# oii, oiii doublet
    mask_width = np.array([50, 100, 100])
    # then set the mask flag (FALSE = masked, weirdly...)
    for i, l in enumerate(lines_to_mask):
        obs['mask'][np.abs(obs['wavelength'] - l) < mask_width[i]] = False
        
    
    print('OBS-- fits file: '+specfolder+objname+'.fits')
        
    # Add unessential bonus info.  This will be stored in output
    obs['objname'] = objname
    obs['z'] = hdulist[2].data['z'][0]
    print('OBS-- redshift '+str(obs['z']))
    obs['apcor'] = row['apcor'] # add in apcor so it's there
    hdulist.close()
    
    print('OBS-- objname: '+obs['objname'])
    
    return obs
    
    
# -----------------
# Helper Functions
# ------------------
    
def set_sdss_lsf(ssp, zred=0.0, objname='', specfolder=None, **extras):
    """Method to make the SSPs have the same (rest-frame) resolution as the
    SDSS spectrographs.  This is only correct if the redshift is fixed, but is
    a decent approximation as long as redshift does not change much.
    """
    print('LSF-- objname: '+objname)
    print('LSF-- zred: '+str(zred))
    
    sdss_filename = specfolder+objname+'.fits'
    sdss_spec, _, _ = load_sdss(sdss_filename)
    wave, delta_v = get_lsf(sdss_spec, zred=zred, **extras)
    assert ssp.libraries[1] == b'miles', "Please change FSPS to the MILES libraries."
    ssp.params['smooth_lsf'] = True
    ssp.set_lsf(wave, delta_v)  
    
def load_sdss(sdss_filename, **extras):
    import astropy.io.fits as pyfits
    with pyfits.open(sdss_filename) as hdus:
        spec = np.array(hdus[1].data)
        info = np.array(hdus[2].data)
        line = np.array(hdus[3].data)
    return spec, info, line
    
def get_lsf(spec, miles_fwhm_aa=2.54, zred=0.0, **extras):
    """This method takes a spec file and returns the quadrature difference
    between the instrumental dispersion and the MILES dispersion, in km/s, as a
    function of wavelength
    """
    lightspeed = 2.998e5  # km/s
    # Get the SDSS instrumental resolution for this plate/mjd/fiber
    wave_obs = 10**spec['loglam']  # observed frame wavelength
    # This is the instrumental velocity resolution in the observed frame
    sigma_v = np.log(10) * lightspeed * 1e-4 * spec['wdisp']
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    # Get the miles velocity resolution function at the corresponding
    # *rest-frame* wavelength
    wave_rest = wave_obs / (1 + zred)
    sigma_v_miles = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
    
    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))
    # Restrict to regions where MILES is used
    good = (wave_rest > 3525.0) & (wave_rest < 7500)

    # return the broadening of the rest-frame library spectra required to match
    # the obserrved frame instrumental lsf
    return wave_rest[good], dsv[good]       
    
##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('gallazzi_05_massmet.txt')

    def __len__(self):
        """ Hack to work with Prospector v0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)
        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])
        
def modified_logsfr_ratios_to_masses_flex(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tquench=None, tflex=None, nflex=None, nfixed=None, 
                                 agebins=None, **extras):
    
    # clip for numerical stability
    nflex = nflex[0]
    nfixed = nfixed[0]
    logsfr_ratio_young = np.clip(logsfr_ratio_young[0], -100, 100)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -100, 100)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -100, 100) # numerical issues...

    # get agebins
    abins = modified_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tquench=tquench, tflex=tflex, nflex=nflex, nfixed=nfixed, **extras)
            
    # if qflag=0, we bonked-- put all the mass in the oldest bin as a 'unfavorable' solution
    if np.array_equal(abins[:-nfixed, 1], np.arange(8, 8+.1*(len(abins)-nfixed), .1)):
        fakemasses = np.zeros(len(agebins))
        fakemasses[-1] = 10**logmass
        return fakemasses       
    
    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
                    #(10**abins[-2:, 1] - 10**abins[-2:, 0])
    old_factor = np.zeros(nfixed)
    for i in range(nfixed): 
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))                    
    # sold_factor = 1. / np.array([np.prod(sold[:i+1]) for i in range(nfixed)])
    # mbin = (10**logmass) / (syoung*dtyoung/dt1 + np.sum(sold_factor*dtold[1:]/dtold[:-1]) + nflex)
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor #sold_factor * mbin * dtold[1:] / dtold[:-1]
    n_masses = np.full(nflex, mbin)

    return np.array([myoung] + n_masses.tolist() + mold.tolist())
    
    ''' 
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum() 
    '''


def modified_logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None, 
                               tquench=None, tflex=None, nflex=None, nfixed=None, **extras):
    """This transforms from SFR ratios to agebins by assuming a constant amount
    of mass forms in each bin agebins = np.array([NBINS,2])

    use equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    
    Edited for new PSB model: youngest bin is 'tquench' long, and it is 
    preceded by 'nflex' young flexible bins, then 'nfixed' older fixed bins
    
    """

    # dumb way to de-arrayify values...
    tquench = tquench[0]
    tflex = tflex[0]
    try: nflex = nflex[0]
    except IndexError: pass
    try: nfixed = nfixed[0]
    except IndexError: pass   
                           
    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)
    
    # flexible time is t_flex - youngest bin (= tquench, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tquench) * 1e9
    
    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))
    
    # if dt1 is very small, we'll get an error that 'agebins must be increasing'
    # to avoid this, anticipate it and return an 'unlikely' solution-- put all the mass 
    # in the first bin. this is complicated by the fact that we CAN'T return two
    # values from this function or FSPS crashes. instead, return a very weirdly
    # specific and exact value that we'll never get to otherwise...
    if dt1 < 1e-4:
        agelims = [0, 8, 8.1]
        for i in range(n_ratio):
            agelims += [agelims[-1]+.1]
        agelims += list(agebins[-nfixed:,1])    
        abins = np.array([agelims[:-1], agelims[1:]]).T
        return abins

    # translate into agelims vector (time bin edges)
    agelims = [1, (tquench*1e9), dt1+(tquench*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += list(10**agebins[-nfixed:,1]) 
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins

def trap(x, y):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.

def massmet_to_logzsol(massmet=None, **extras):
    '''Simple function to get metallicity from massmet. '''
    return massmet[1]   
    
def massmet_to_logmass(massmet=None, **extras):
    '''Another simple function to get mass from massmet.
    This might need to be more complicated with how masses works? ''' 
    return massmet[0]       
    

def massmet_to_masses(massmet=None, logsfr_ratios=None, agebins=None, 
                        tquench=None, tflex=None, nflex=None, nfixed=None, 
                        logsfr_ratio_young=None, logsfr_ratio_old=None, **extras):
    '''Given the total mass, sfr ratios, and agebins, compute the mass in each bin '''                             
    
    masses = modified_logsfr_ratios_to_masses_flex(logmass=massmet[0], 
        logsfr_ratios=logsfr_ratios, logsfr_ratio_young=logsfr_ratio_young,
        logsfr_ratio_old=logsfr_ratio_old,agebins=agebins, tquench=tquench, 
        tflex=tflex, nflex=nflex, nfixed=nfixed)
        
    return masses    
    
    
def umachine_priors(agebins=None, zred=None, tquench=None,
                     tflex=None, nflex=None, nfixed=None, **extras):
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

    # get mass in each bin
    myoung = np.sum(mass[(newages >= abins_age[0,0]) & (newages <= abins_age[0,1])])
    mold = []
    for i in range(nflex+1, len(abins_age)):
        mold.append(np.sum(mass[(newages >= abins_age[i,0]) & (newages <= abins_age[i,1])]))
    mflex = np.sum(mass[(newages >= abins_age[1,0]) & (newages <= abins_age[nflex,1])])    

    # adjust agebins according to mflex
    # each of the nflex flex bins should have (mflex / nflex) mass
    idx = (newages >= abins_age[1,0]) & (newages <= abins_age[nflex,1]) # part of flex range
    agelims_flex = []
    for i in range(nflex):
        agelims_flex.append(np.interp(mflex/nflex*i, np.cumsum(mass[idx]), newages[idx]))
    abins_age[1:nflex, 1] = agelims_flex[1:]
    abins_age[2:nflex+1, 0] = agelims_flex[1:]

    # remake agebins
    agebins = np.log10(1e9*abins_age)

    # now get the sfr in each bin
    sfrs = np.zeros((len(abins_age)))
    masses = np.zeros((len(abins_age)))
    for i in range(len(sfrs)):
        # relevant umachine ages
        idx = (newages >= abins_age[i,0]) & (newages <= abins_age[i,1])
        sfrs[i] = trap(newages[idx], sfh[idx]) / (abins_age[i,1] - abins_age[i,0])
        masses[i] = np.sum(mass[idx])

    # young is easy
    logsfr_ratio_young = np.log10(sfrs[0] / sfrs[1])

    # old is w ref to the oldest flex bin
    logsfr_ratio_old = np.ones(nfixed)
    for i in range(nfixed):
        logsfr_ratio_old[i] = sfrs[nflex+i] / sfrs[nflex+i+1]
    logsfr_ratio_old = np.log10(logsfr_ratio_old)    

    # and finally the flex bins
    logsfr_ratios = np.ones(nflex-1)
    for i in range(nflex-1):
        logsfr_ratios[i] = sfrs[i+1] / sfrs[i+2] 
    logsfr_ratios = np.log10(logsfr_ratios)
    
    return(agebins, logsfr_ratio_young, logsfr_ratios, logsfr_ratio_old, logmass)
    
    
# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    
    # add in the lsf
    set_sdss_lsf(sps.ssp, **extras)
                       
    return sps

    
# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None   


def prior_transform(u):        
    return model.prior_transform(u)

    
# -----------------
# Noise Model
# ------------------    
def build_noise(**extras):
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None 
    
        
    
# -----------------
# SED Model
# ------------------    

def load_model(zred = 0.0, fixed_metallicity=None, add_dust=False,
               add_neb=True, luminosity_distance=None, agelims=None, objname=None,
               catfile=None, binmax=None, tquench=None, 
               tflex=None, nflex=None, nfixed=None, **extras):
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
               
    print('MODEL-- zred: '+str(zred))
               
    # --- Use the continuity SFH template. ---
    model_params = TemplateLibrary["continuity_flex_sfh"]
    
    # Now we add smoothing parameters
    # This accounts for the resolution of the spectrum: convolve model to match data resolution, 
    #   fit for velocity dispersion
    model_params.update(TemplateLibrary["spectral_smoothing"])
    
    # test this: optimize out moderate-order polynomial at every likelihood call
    model_params.update(TemplateLibrary['optimize_speccal'])

    # set the redshift
    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = zred
    
    # set IMF to chabrier (default is kroupa)
    model_params['imf_type']['init'] = 1
    
    # set SFH bins
    # last agebin is 0.1 Gyr wide
        # first one is joel's default 30 Myr
    tuniv = cosmo.age(zred).value
    agelims = np.array([1, tquench*1e9] + \
        np.linspace((tquench+.1)*1e9, (tflex)*1e9, nflex).tolist() \
        + np.linspace(tflex*1e9, tuniv*1e9, nfixed+1)[1:].tolist())
    agebins = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T     
    ncomp = agebins.shape[0] 
    
    # now, use universe machine SFHs to calculate better priors for 
    # logsfr_ratios, agebins
    um_agebins, um_logsfr_ratio_young, um_logsfr_ratios, um_logsfr_ratio_old, um_logmass = \
        umachine_priors(agebins, zred, tquench, tflex, nflex, nfixed)
        
    # load nvariables and agebins
    model_params['mass']['N'] = ncomp
    model_params['mass']['init'] = np.full(ncomp,1e6)
    model_params['mass']['depends_on'] = massmet_to_masses
    model_params['mass']['transform'] = massmet_to_masses
    model_params['agebins']['N'] = ncomp
    model_params['agebins']['init'] = um_agebins
    model_params['agebins']['depends_on'] = modified_logsfr_ratios_to_agebins
    
    # log_sfr parameters
    model_params["logsfr_ratios"]["N"] = nflex - 1
    model_params["logsfr_ratios"]["init"] = um_logsfr_ratios #np.zeros(nflex-1) 
    model_params["logsfr_ratios"]["prior"] = priors.StudentT(mean=um_logsfr_ratios, #np.zeros(nflex-1), 
        scale = np.ones(nflex-1)*0.3, df = np.ones(nflex-1))    
    model_params['logsfr_ratio_old']['N'] = nfixed
    model_params["logsfr_ratio_old"]["init"] = um_logsfr_ratio_old #np.zeros(nfixed) 
    model_params["logsfr_ratio_old"]["prior"] = priors.StudentT(mean=um_logsfr_ratio_old, #np.zeros(nfixed), 
        scale = np.ones(nfixed)*0.3, df = np.ones(nfixed))
    model_params['logsfr_ratio_young']['prior'] = priors.StudentT(mean=um_logsfr_ratio_young-1, #0,
        scale=.7, df=1)    
                
    # how long has galaxy been quenched? e.g., what's length of last agebin
    model_params['tquench'] = {'name':'tquench', 'N':1, 'isfree':True,
        'init':.2, 'prior':priors.TopHat(mini=.01, maxi=1.5)}   
        
    model_params['tflex'] = {'name':'tflex', 'N':1, 'isfree': False, 'init':tflex}          
    model_params['nflex'] = {'name':'nflex', 'N':1, 'isfree': False, 'init':nflex}          
    model_params['nfixed'] = {'name':'nfixed', 'N':1, 'isfree': False, 'init':nfixed}                                                             
    
    # following joel's email-- change mass units to mformed
    model_params['mass_units'] = {'name': 'mass_units', 'N': 1,
                          'isfree': False,
                          'init': 'mformed'}

    # massmet controls total mass and metallicity
    model_params['massmet'] = {'name':'massmet', 'N':2, 'isfree':True, 'init':[um_logmass,0],
                        'prior':MassMet(z_mini=-0.5, z_maxi=1.0, mass_mini=9.5, mass_maxi=12.5)}
                        
    
    # default includes a free 'logmass' -- update it based on massemt
    model_params['logmass'] = {'N':1, 'depends_on':massmet_to_logmass, 'isfree':False,
                          'init':model_params['massmet']['init'][0]}

    # metallicity-- depends on massmet prior
    model_params['logzsol'] = {'N':1, 'depends_on':massmet_to_logzsol, 'isfree':False, 
        'init':model_params['massmet']['init'][1]}
    
    # dust: test out a version with kc03!    # use calzetti law
    # model_params['dust_type'] = {'N':1, 'isfree':False, 'init':2}
    model_params['dust_type'] = {'N':1, 'isfree':False, 'init':4}
    model_params['dust_index'] = {'N':1, 'isfree':True, 'init':0,
        'prior':priors.TopHat(mini=-1, maxi=0.4)}
    model_params['dust1'] = {'N':1, 'isfree':False, 'init':0.0} # have to set this to 0 for calzetti
    model_params['dust2']['init'] = 0.5 / (2.5*np.log10(np.e))
    model_params['dust2']['prior'] = priors.TopHat(mini=0.0,
          maxi=2.5 / (2.5*np.log10(np.e))) # factor is for AB magnitudes -> optical depth (v small correction...)
    
    # change the prior on sigma_smooth depending on what the ppxf sigma is
    sigmacat = np.load(catfile, allow_pickle=True)
    try:
        row = sigmacat[sigmacat['specname'] == objname+'.fits'][0]
        model_params['sigma_smooth']['init'] = row['sigma']
        # model_params['sigma_smooth']['prior'] = priors.TopHat(mini=row['sigma'] - row['sigma_err'] - 75,
        #     maxi=row['sigma'] + row['sigma_err'] + 75)
        model_params['sigma_smooth']['prior'] = priors.Normal(mean=row['sigma'], sigma=row['sigma_err'])
        print('MODEL-- set sigma to '+'{:.0f}'.format(row['sigma'])+' km/s, gaussian prior')    
    except IndexError:
        # this means that this is the one alma galaxy that got cut out of the final sample...
        model_params['sigma_smooth']['init'] = 200
        model_params['sigma_smooth']['prior'] = priors.TopHat(mini=100, maxi=300)
        print('MODEL-- no sigma in catalog, set to 200 km/s with tophat prior') 
    
    # add in spectroscopic jitter term (re joel's email 6/10/19)    
    model_params['spec_jitter'] = {"N": 1, "isfree": True, "init": 1.0, 
        "prior": priors.TopHat(mini=1.0, maxi=1.5)} # 15.0)}    
        
    # pixel outlier model (from joel)
    model_params['f_outlier_spec'] = {"N": 1, 
                                      "isfree": True, 
                                      "init": 0.01,
                                      "prior": priors.TopHat(mini=0, maxi=0.5)}
    model_params['nsigma_outlier_spec'] = {"N": 1, 
                                          "isfree": False, 
                                          "init": 5.0}  

    # # Change the model parameter specifications based on some keyword arguments
    if add_dust:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # # Now instantiate the model using this new dictionary of parameter specifications
    # model = SedModel(model_params)
    
    # for now, test a version with polynomial optimization
    print('using polynomial optimization model')
    model = PolySedModel(model_params)
    
    print('MODEL-- redshift: '+str(model_params['zred']['init']))

    return model    
            
    
    
    
    

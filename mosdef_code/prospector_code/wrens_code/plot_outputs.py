import numpy as np
from prospect.models import priors, SedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
import sedpy
from astropy.io import fits
from scipy import signal
import dynesty
import h5py
import matplotlib.pyplot as plt
plt.interactive(True)
from astropy.cosmology import FlatLambdaCDM
import cornerplot
from prospect.models.transforms import logsfr_ratios_to_masses
import sys

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

from prospect.io.read_results import results_from, get_sps
from prospect.io.read_results import traceplot, subcorner

if len(sys.argv) > 0:
	outroot = sys.argv[1]
else:
	outroot = 'simpleSFH_J2202-0033_1558375030_mcmc.h5'
nonpar = True
print('Making plots for '+outroot)

# functions from tom to get theta values for different percentiles
def quantile(data, percents, weights=None):
	''' percents in units of 1%
	weights specifies the frequency (count) of data.
	'''
	if weights is None:
		return np.percentile(data, percents)
	ind = np.argsort(data)
	d = data[ind]
	w = weights[ind]
	p = 1.*w.cumsum()/w.sum()*100
	y = np.interp(percents, p, d)
	return y

def get_percentiles(res,mod, ptile=[16, 50, 84], start=0.0, thin=1, **extras):
	"""Get get percentiles of the marginalized posterior for each parameter.

	:param res:
		A results dictionary, containing a "chain" and "theta_labels" keys.

	:param ptile: (optional, default: [16, 50, 84])
	   A list of percentiles (integers 0 to 100) to return for each parameter.

	:param start: (optional, default: 0.5)
	   How much of the beginning of chains to throw away before calculating
	   percentiles, expressed as a fraction of the total number of iterations.

	:param thin: (optional, default: 10.0)
	   Only use every ``thin`` iteration when calculating percentiles.

	:returns pcts:
	   Dictionary with keys giving the parameter names and values giving the
	   requested percentiles for that parameter.
	"""

	parnames = np.array(res.get('theta_labels', mod.theta_labels()))
	niter = res['chain'].shape[-2]
	start_index = np.floor(start * (niter-1)).astype(int)
	if res["chain"].ndim > 2:
		flatchain = res['chain'][:, start_index::thin, :]
		dims = flatchain.shape
		flatchain = flatchain.reshape(dims[0]*dims[1], dims[2])
	elif res["chain"].ndim == 2:
		flatchain = res["chain"][start_index::thin, :]
	pct = np.array([quantile(p, ptile, weights=res.get("weights", None)) for p in flatchain.T])
	return dict(zip(parnames, pct)) 
	
# grab results (dictionary), the obs dictionary, and our corresponding models
res, obs, mod = results_from("{}".format(outroot), dangerous=True)
# it doesn't make a model-- load by hand
# mod = load_model(**res["run_params"])
sps = get_sps(res)
print('Loaded results')


# traceplot
tracefig = traceplot(res, figsize=(10,5))
plt.savefig('savio/plots/toymodel/zinf/trace/'+obs['objname']+'.pdf', bbox_inches='tight')
plt.close()

# corner plot
# maximum a posteriori (of the locations visited by the MCMC sampler)
imax = np.argmax(res['lnprobability'])
theta_max = res['chain'][imax, :].copy()
print('MAP value: {}'.format(theta_max))
# We throuw out the first 500 samples because they are (usually) very low probability 
# and can throw off the plotting algorithms
# cornerfig = subcorner(res, start=0, thin=1, fig=plt.subplots(len(theta_max), len(theta_max),figsize=(7,7))[0])
fig, axes = plt.subplots(len(theta_max), len(theta_max), figsize=(7,7))
axes = cornerplot.allcorner(res['chain'].T, mod.theta_labels(), axes, show_titles=True, 
	span=[0.997]*len(mod.theta_labels()), weights=res.get("weights", None))
fig.savefig('savio/plots/toymodel/zinf/corner/'+obs['objname']+'.pdf', bbox_inches='tight')  
plt.close(fig)	    
print('Made cornerplot')

# look at sed & residuals
# generate models
mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps)
mass = mod.params['mass'] # get this now before re-running a new model
# wavelength vectors
a = 1.0 + mod.params.get('zred', 0.0) # cosmological redshifting
# photometric effective wavelengths
wphot = np.array(obs["phot_wave"])
# spectroscopic wavelengths
if obs["wavelength"] is None:
	# *restframe* spectral wavelengths, since obs["wavelength"] is None
	wspec = sps.wavelengths.copy()
	wspec *= a #redshift them
else:
	wspec = obs["wavelength"]

# get real 16/50/84% spectra
# only calculate from 1000 highest-weight samples
print('Starting to calculate spectra...')
weights = res.get('weights',None)
idx = np.argsort(weights)[-1000:]
allspec = np.zeros((len(mspec_map), len(idx)))
allmfrac = np.zeros((len(idx)))
for ii, i in enumerate(idx):
	allspec[:,ii], a, allmfrac[ii] = mod.mean_model(res['chain'][i,:], obs, sps=sps)
spec16 = np.array([quantile(allspec[i,:], 16, weights = weights[idx]) for i in range(allspec.shape[0])])
spec50 = np.array([quantile(allspec[i,:], 50, weights = weights[idx]) for i in range(allspec.shape[0])])
spec84 = np.array([quantile(allspec[i,:], 84, weights = weights[idx]) for i in range(allspec.shape[0])])
print('Done calculating spectra')

# Make plot of data and model
c = 2.99792458e18
fig, ax = plt.subplots(3,1,figsize=(8,10))
ax[0].plot(wspec, mspec_map*c/wspec**2., label='Model spectrum (MAP)',
	   lw=1.5, color='green', alpha=0.7, zorder=10)	   
ax[0].errorbar(wphot, mphot_map*c/wphot**2., label='Model photometry (MAP)',
		 marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
		 markerfacecolor='none', markeredgecolor='green', 
		 markeredgewidth=3)
ax[0].errorbar(wphot, obs['maggies']*c/wphot**2, yerr=obs['maggies_unc']*c/wphot**2, 
		 label='Observed photometry', ecolor='red', 
		 marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
		 markerfacecolor='none', markeredgecolor='black', 
		 markeredgewidth=3)			   
ax[0].plot(wspec, obs['spectrum']*c/wspec**2, color='black', lw=.75, label='Observed spectrum', zorder=1)
ax[0].fill_between(wspec, (obs['spectrum'] - obs['unc'])*c/wspec**2, 
	(obs['spectrum'] + obs['unc'])*c/wspec**2, color='grey', alpha=.5, zorder=0)
ax[0].fill_between(wspec, spec16*c/wspec**2, spec84*c/wspec**2, color='green', alpha=.4, zorder=1)
norm_wl = ((wspec>6300) & (wspec<6500))
norm = np.median(obs['spectrum'][norm_wl]*c/wspec[norm_wl]**2)
ax[0].set_ylim((-norm*.5, norm*3))
ax[0].set_xlabel('Wavelength [A]')
ax[0].set_ylabel(r'$F_{\lambda}$')
ax[0].legend(loc='best', fontsize=14)
ax[0].set_title(outroot)
print('Made spectrum plot')

######################## SFH for FLEXIBLE continuity model ########################
from squiggle_flex_continuity import modified_logsfr_ratios_to_masses_flex, modified_logsfr_ratios_to_agebins

# actual sfh percentiles
flatchain = res["chain"]
start = .5
niter = res['chain'].shape[-2]
start_index = np.floor(start * (niter-1)).astype(int)
allsfrs = np.zeros((flatchain.shape[0], len(mod.params['agebins'])))
masscum = np.zeros_like(allsfrs)
 #len(flatchain[0,mod.theta_index["logsfr_ratios"]])+1))
allagebins = np.zeros((flatchain.shape[0], len(mod.params['agebins']), 2))
for iteration in range(flatchain.shape[0]):
	logr = flatchain[iteration, mod.theta_index["logsfr_ratios"]]
	tquench = flatchain[iteration, mod.theta_index['tquench']]
	logr_young = flatchain[iteration, mod.theta_index['logsfr_ratio_young']]
	logr_old = flatchain[iteration, mod.theta_index['logsfr_ratio_old']]
	try:
		logmass = flatchain[iteration, mod.theta_index['massmet']][0] #flatchain[iteration, mod.theta_index["logmass"]]
	except:
		logmass = flatchain[iteration, mod.theta_index["logmass"]]		
	agebins = modified_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], 
		tquench=tquench, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
	allagebins[iteration, :] = agebins
	dt = 10**agebins[:, 1] - 10**agebins[:, 0]
	masses = modified_logsfr_ratios_to_masses_flex(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
		logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
		tquench=tquench, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
	allsfrs[iteration,:] = (masses	/ dt)
	masscum[iteration,:] = np.cumsum(masses) / np.sum(masses)
sfrmap = allsfrs[imax,:]

# to calculate quantiles on SFR, first have to put everything on the same timebin scale
# for simplicity, just make this a 0.1Gyr wide bin
from scipy.interpolate import interp1d
tuniv = cosmo.age(mod.params['zred'][0]).value
allagebins_ago = 10**allagebins/1e9
tflex = mod.params['tflex'][0]
age_interp = np.append(np.arange(0,tflex,.01),allagebins_ago[1000,:,:].flatten()[-5:])
age_interp[0] = 1e-9
allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
masscum_interp = np.zeros_like(allsfrs_interp)
for i in range(flatchain.shape[0]):
	f = interp1d(allagebins_ago[i,:].flatten(), np.repeat((allsfrs[i,:]),2))
	allsfrs_interp[i,:] = f(age_interp)
	f = interp1d(allagebins_ago[i,:].flatten(), np.repeat((masscum[i,:]),2))
	masscum_interp[i,:] = f(age_interp)
	# allsfrs_interp[i,:] = np.interp(x=age_interp, xp=allagebins_ago[i,:].flatten(),
	#	  fp=np.repeat(allsfrs[i,:], 2))
# sfr percentiles	 
sfr16 = np.array([quantile(allsfrs_interp[:,i], 16, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])]) 
sfr50 = np.array([quantile(allsfrs_interp[:,i], 50, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])])		   
sfr84 = np.array([quantile(allsfrs_interp[:,i], 84, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])])		   
# cumulative mass percentiles
mass16 = 1-np.array([quantile(masscum_interp[:,i], 16, weights=res.get('weights', None)) for i in range(masscum_interp.shape[1])]) 
mass50 = 1-np.array([quantile(masscum_interp[:,i], 50, weights=res.get('weights', None)) for i in range(masscum_interp.shape[1])])		   
mass84 = 1-np.array([quantile(masscum_interp[:,i], 84, weights=res.get('weights', None)) for i in range(masscum_interp.shape[1])])			   

# plot sfh and percentiles
# ax[1].fill_between(agebins_ago.flatten(), np.repeat((sfr16),2), np.repeat((sfr84),2), color='grey', alpha=.5)
ax[1].fill_between(age_interp, sfr16, sfr84, color='grey', alpha=.5)
# ax[1].plot(allagebins_ago[imax,:].flatten(), np.repeat((sfrmap),2), color='black', lw=2)
ax[1].plot(age_interp, sfr50, color='black', lw=1.5)
# ax[1].plot(agebins_ago.flatten(), np.repeat((sfr50),2), color='grey', lw=1.5)
# ax[1].set_xlim((14,tuniv-.1))
ax[1].set_xlim((tuniv+.1,-.1))
# ax[1].set_yscale('log')
ax[1].set_ylabel('SFR [Msun/yr]')
ax[1].set_xlabel('years before observation [Gyr]')

# cumulative mass fraction plot
ax[2].fill_between(age_interp, mass16, mass84, color='grey', alpha=.5)
ax[2].plot(age_interp, mass50, color='black', lw=1.5)
ax[2].set_xlim((tuniv+.1,-.1))
ax[2].set_ylabel('Cumulative mass fraction')
ax[2].set_xlabel('years before observation [Gyr]')

fig.savefig('savio/plots/toymodel/zinf/sfh/'+obs['objname']+'.pdf', bbox_inches='tight')
plt.close(fig)

print('Made SFH plot')

# ######################## SFH for continuity model ########################
# # actual sfh percentiles
# flatchain = res["chain"]
# start = .5
# niter = res['chain'].shape[-2]
# start_index = np.floor(start * (niter-1)).astype(int)
# allsfrs = np.zeros((flatchain.shape[0], len(flatchain[0,mod.theta_index["logsfr_ratios"]])+1))
# agebins = mod.params["agebins"]
# dt = 10**agebins[:, 1] - 10**agebins[:, 0]
# for iteration in range(flatchain.shape[0]):
#	  logr = flatchain[iteration, mod.theta_index["logsfr_ratios"]]
#	  try:
#		  logmass = flatchain[iteration, mod.theta_index['massmet']][0] #flatchain[iteration, mod.theta_index["logmass"]]
#	  except:
#		  logmass = flatchain[iteration, mod.theta_index["logmass"]]
#	  masses = logsfr_ratios_to_masses(logsfr_ratios=logr, logmass=logmass, agebins=agebins)
#	  allsfrs[iteration,:] = (masses  / dt)
# sfr16 = np.array([quantile(allsfrs[:,i], 16,weights=res.get("weights", None)) for i in range(allsfrs.shape[1])])
# sfr50 = np.array([quantile(allsfrs[:,i], 50,weights=res.get("weights", None)) for i in range(allsfrs.shape[1])])
# sfr84 = np.array([quantile(allsfrs[:,i], 84,weights=res.get("weights", None)) for i in range(allsfrs.shape[1])])
# sfrmap = allsfrs[imax,:]
#
# tuniv = cosmo.age(mod.params['zred'][0]).value
# # agebins_lookback = (cosmo.age(0).value - tuniv) + (10**agebins) / 1e9
# agebins_ago = 10**agebins/1e9 # time since "now"
#
# # burst start and end time (this should be made more sophisticated)
# burststart = agebins_ago[::-1,0][np.where(sfr50[::-1][1:] / sfr50[::-1][:-1] > 2)][0]
# burstend = agebins_ago[::-1,0][np.where(sfr50[::-1][1:] / sfr50[::-1][:-1] < .1)][-1]
#
# # plot sfh and percentiles
#
# ax[1].fill_between(agebins_ago.flatten(), np.repeat((sfr16),2), np.repeat((sfr84),2), color='grey', alpha=.5)
# ax[1].plot(agebins_ago.flatten(), np.repeat((sfrmap),2), color='black', lw=2)
# ax[1].plot(agebins_ago.flatten(), np.repeat((sfr50),2), color='grey', lw=1.5)
# # ax[1].set_xlim((14,tuniv-.1))
# ax[1].set_xlim((tuniv+.1,-.1))
# ax[1].set_yscale('log')
# ax[1].set_ylabel('SFR [Msun/yr]')
# ax[1].set_xlabel('years before observation [Gyr]')
# # ax[1].axvline(burststart, ls='dashed', color='green')
# # ax[1].axvline(burstend, ls='dashed', color='firebrick')


# ######################## SFH for FLEXIBLE continuity model ########################
# from squiggle_flex_continuity_nomassmet import modified_logsfr_ratios_to_masses_flex, modified_logsfr_ratios_to_agebins
#
# # actual sfh percentiles
# flatchain = res["chain"]
# start = .5
# niter = res['chain'].shape[-2]
# start_index = np.floor(start * (niter-1)).astype(int)
# allsfrs = np.zeros((flatchain.shape[0], len(mod.params['agebins']))) #len(flatchain[0,mod.theta_index["logsfr_ratios"]])+1))
# allagebins = np.zeros((flatchain.shape[0], len(mod.params['agebins']), 2))
# for iteration in range(flatchain.shape[0]):
#	  logr = flatchain[iteration, mod.theta_index["logsfr_ratios"]]
#	  logr_young = flatchain[iteration, mod.theta_index['logsfr_ratio_young']]
#	  logr_old = flatchain[iteration, mod.theta_index['logsfr_ratio_old']]
#	  try:
#		  logmass = flatchain[iteration, mod.theta_index['massmet']][0] #flatchain[iteration, mod.theta_index["logmass"]]
#	  except:
#		  logmass = flatchain[iteration, mod.theta_index["logmass"]]
#	  agebins = modified_logsfr_ratios_to_agebins(logr, mod.params['agebins'])
#	  allagebins[iteration, :] = agebins
#	  dt = 10**agebins[:, 1] - 10**agebins[:, 0]
#	  masses = modified_logsfr_ratios_to_masses_flex(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
#		  logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old)
#	  allsfrs[iteration,:] = (masses	/ dt)
# sfrmap = allsfrs[imax,:]
#
# # to calculate quantiles on SFR, first have to put everything on the same timebin scale
# # for simplicity, just make this a 0.1Gyr wide bin
# from scipy.interpolate import interp1d
# tuniv = cosmo.age(mod.params['zred'][0]).value
# allagebins_ago = 10**allagebins/1e9
# age_interp = np.append(np.arange(0,tuniv,.01),tuniv)
# age_interp[0] = 1e-9
# allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
# for i in range(flatchain.shape[0]):
#	  f = interp1d(allagebins_ago[i,:].flatten(), np.repeat((allsfrs[i,:]),2))
#	  allsfrs_interp[i,:] = f(age_interp)
#	  # allsfrs_interp[i,:] = np.interp(x=age_interp, xp=allagebins_ago[i,:].flatten(),
#	  #		fp=np.repeat(allsfrs[i,:], 2))
# sfr16 = np.array([quantile(allsfrs_interp[:,i], 16, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])])
# sfr50 = np.array([quantile(allsfrs_interp[:,i], 50, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])])
# sfr84 = np.array([quantile(allsfrs_interp[:,i], 84, weights=res.get('weights', None)) for i in range(allsfrs_interp.shape[1])])
#
#
# # plot sfh and percentiles
# # ax[1].fill_between(agebins_ago.flatten(), np.repeat((sfr16),2), np.repeat((sfr84),2), color='grey', alpha=.5)
# ax[1].fill_between(age_interp, sfr16, sfr84, color='grey', alpha=.5)
# ax[1].plot(allagebins_ago[imax,:].flatten(), np.repeat((sfrmap),2), color='black', lw=2)
# ax[1].plot(age_interp, sfr50, color='grey', lw=1.5)
# # ax[1].plot(agebins_ago.flatten(), np.repeat((sfr50),2), color='grey', lw=1.5)
# # ax[1].set_xlim((14,tuniv-.1))
# ax[1].set_xlim((tuniv+.1,-.1))
# ax[1].set_yscale('log')
# ax[1].set_ylabel('SFR [Msun/yr]')
# ax[1].set_xlabel('years before observation [Gyr]')
#
# print('Made SFH plot')

# f.savefig('Plots/'+outroot+'.pdf', bbox_inches='tight')


# # if nonparametric, also plot SFH
# if not nonpar:
#	raise SystemExit
#
# sfrs = mass / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0]) # in Msun/yr
# sfrs16 = mass16 / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0]) # in Msun/yr
# sfrs84 = mass84 / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0]) # in Msun/yr
# agebins = np.insert(mod.params['agebins'][:,1], 0, mod.params['agebins'][0,0]) # log(yr)
# agebins_gyr = 10**agebins / 1e9 # Gyr
# centers = (agebins_gyr[1:] + agebins_gyr[:-1]) / 2 # Gyr
# widths = agebins_gyr[1:] - agebins_gyr[:-1] # Gyr
# tuniv = cosmo.age(mod.params['zred'][0]).value
#
# plt.figure()
# plt.bar(centers, sfrmap, width=widths, edgecolor=['black']*len(sfrs), fill=False, lw=2)
# plt.bar(centers, sfr16, width=widths, edgecolor=['grey']*len(sfrs), fill=False, lw=2)
# plt.bar(centers, sfr84, width=widths, edgecolor=['grey']*len(sfrs), fill=False, lw=2)
# plt.ylabel('SFR [Msun/yr]')
# plt.xlabel('time [Gyr]')
# plt.axvline(tuniv, color='black', ls='dashed', label='age of universe at redshift of galaxy')
# plt.text(tuniv-1.2, np.log10(sfrs).max()/1.5, 'age of\nuniverse', fontsize=12, color='black')
#
#
#
# # plot w/ seds and sfhs for each conf interval
# f, ax = plt.subplots(3,2,figsize=(10,10), sharex='col', sharey='col')
# f.subplots_adjust(hspace=0.1)
# mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps)
# sfr = mod.params['mass'] / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0])
# ax[0,0].plot(wspec, mspec_map, lw=1.5, color='black', zorder=10)
# ax[0,1].bar(centers, sfr, width=widths, color='black')
# spec, phot, _ = mod.mean_model(pct16, obs, sps=sps)
# sfr = mod.params['mass'] / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0])
# ax[1,0].plot(wspec, spec, lw=1.5, color='black', zorder=10)
# ax[1,1].bar(centers, sfr, width=widths, color='black')
# spec, phot, _ = mod.mean_model(pct84, obs, sps=sps)
# sfr = mod.params['mass'] / (10**mod.params['agebins'][:,1] - 10**mod.params['agebins'][:,0])
# ax[2,0].plot(wspec, spec, lw=1.5, color='black', zorder=10)
# ax[2,1].bar(centers, sfr, width=widths, color='black')
# ax[2,0].set_xlabel('Wavelength')
# ax[1,0].set_ylabel('Flux (maggies)')
# ax[1,1].set_ylabel('SFR')
# ax[2,1].set_xlabel('age of universe (Gyr)')
# ax[0,0].text(4000,4.5e-8,'MAP',fontsize=14)
# ax[1,0].text(4000,4.5e-8,'16th percentile',fontsize=14)
# ax[2,0].text(4000,4.5e-8,'84th percentile',fontsize=14)


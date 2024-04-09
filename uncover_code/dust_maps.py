import numpy as np
import matplotlib.pyplot as plt; plt.interactive(True)
from matplotlib.colors import LogNorm
from astropy.io import fits
from glob import glob
from astropy.wcs import WCS
import astropy.visualization as viz
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)
from matplotlib.backends.backend_pdf import PdfPages
from plot_log_linear_rgb import make_linear_rgb
from scipy.signal import medfilt
import sedpy
from plot_log_linear_rgb import make_linear_rgb, make_log_rgb

# sys.path.append('../uncover-morphologeurs/')
# from fit_pysersic import fit_pysersic
import seaborn as sns
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', 
    "lines.linewidth":1.2, "figure.figsize":(5,5)})

transform = viz.LogStretch(a=1500) #+ viz.ManualInterval()

lines = np.array([[r'3.3$\mu$m PAH',33000], [r'Pa$\alpha$',18750], [r'Pa$\beta$',12820], [r'HeI',10830],[r'[SIII]',9069],
    [r'H$\alpha$+[NII]',6563], [r'H$\beta$+[OIII]',4863], [r'[OII]',3700], #['Balmer break',3645], 
    [r'Ly$\alpha$',1215]])
filter_wavelengths = {'F140M':[1.331, 1.479], 'F162M':[1.542, 1.713], 'F182M':[1.722, 1.968], 'F210M':[1.992, 2.201],
    'F250M':[2.412, 2.595], 'F300M':[2.831, 3.157], 'F335M':[3.177, 3.537],
    'F360M':[3.426, 3.814], 'F410M':[3.866, 4.302], 'F430M':[4.167, 4.398], 
    'F460M':[4.515, 4.747], 'F480M':[4.662,4.973]}
    

def _rebin(arr, new_2dshape):
    shape = (new_2dshape[0], arr.shape[0] // new_2dshape[0],
             new_2dshape[1], arr.shape[1] // new_2dshape[1])
    return arr.reshape(shape).sum(-1).sum(-2)

def get_cutout(x, y, dt, im):
    """
    make a cutout of an image "im" centered at pixel x/y with size dt
    """
    il = np.maximum(0,int(y)-(dt+1))
    iu = np.minimum(int(y)+(dt+1), im.shape[0])
    jl = np.maximum(0,int(x)-(dt+1))
    ju = np.minimum(int(x)+(dt+1), im.shape[1])
    return im[il:iu, jl:ju]

def get_all_cutouts(ra, dec, dt=50):
    allcuts = []
    for f in filters:  
        x, y = wcs[f].wcs_world2pix([ra], [dec], 1)  
        # if f in filters_sw:
        #     bigcut = get_cutout(x[0], y[0], dt*2+1, images[f])
        #     binned = _rebin(bigcut, (int(bigcut.shape[0]/2),
        #                                     int(bigcut.shape[1]/2)))
        #     allcuts.append(binned)
        # else:
        allcuts.append(get_cutout(x[0], y[0], dt, images[f]))
    return np.array(allcuts)    
    
    
# load spec-z catalog
speczs = np.genfromtxt('/Users/wren/Downloads/v2_speczs.csv', names=True, usecols=(0,1,2,3,4,5,6,7,8), delimiter=',')    
speczs['Average_rating'][np.isfinite(speczs['Override_rating'])] = speczs['Override_rating'][np.isfinite(speczs['Override_rating'])]
speczs[speczs['Average_rating']>=2]

# ehre we want to do maps of both paschen-b and Ha
highz = 4.8 / 1.282 - 1
lowz = 1.4 / .6563 - 1
sample = speczs[(speczs['z']<=highz) & (speczs['z']>=lowz)]  
    
	
# load catalog
with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/v0.7/UNCOVER_v3.0.2_LW_SUPER_SPScatalog_spsv1.2.fits') as hdu:
    spscat = hdu[1].data  
	
# john first-look catalogs!!
with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/catalogs/v5.0.1/UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits') as hdu:
    mbphot = hdu[1].data   
with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/catalogs/v5.0.1/UNCOVER_v5.0.1_LW_SUPER_CATALOG_sfhz_PHOTOZ.fits') as hdu:
    mbzout = hdu[1].data 
    
# # open segmap
# with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/v0.7/UNCOVER_v3.0.0_LW_SEGMAP.fits') as hdu:
#     segmap = hdu[0].data
#     segWCS = WCS(hdu[0].header)
#
# # get psf
# with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/v0.7/PSF/f444w_psf.fits') as hdu:
#     psf_f444w = hdu[0].data
# if np.min(psf_f444w) < 0:
#     print('PSF has negative values...')
#     psf_f444w[psf_f444w<0] = 0
# if np.abs( np.sum(psf_f444w) - 1) > 0.1:
#     print("Normalizing PSF to 1")
#     psf_f444w = psf_f444w/np.sum(psf_f444w)  

# load in all our images and WCSs    
images = {}
wcs = {}
photflam = {}
photplam = {}
filters = np.array(['f070w', 'f090w', 'f140m', 'f150w', 'f162m', 'f182m', 'f210m', 'f250m', 'f300m', 'f335m', 'f360m', 'f410m', 'f430m', 'f460m', 'f480m'])
# filters = np.array(['f070w', 'f090w', 'f115w', 'f140m', 'f150w', 'f162m', 'f182m', 'f200w', 'f210m', 'f250m', 'f277w', 'f300m', 'f335m', 'f360m', 'f356w', 'f410m', 'f430m', 'f444w', 'f460m', 'f480m'])
mb_filters = np.array([f for f in filters if f.endswith('m')])
filters_sw = filters[:9] # these have 0.02" pixel scale (LW = 0.04")
for f in filters:
    fname = glob('/Volumes/DarkPhoenix/Surveys/UNCOVER/v7.2/bcg_subtracted/uncover_v7.2_abell2744clu_*'+f+'*.fits')
    if len(fname)!=1:
        print('nothing for '+f)
        continue
    with fits.open(fname[0]) as hdu:
        images[f] = hdu[0].data
        wcs[f] = WCS(hdu[0].header)
        photflam[f] = hdu[0].header['PHOTFLAM']
        photplam[f] = hdu[0].header['PHOTPLAM']

# get MB filter objects        
# fils = sedpy.observate.load_filters([fil for fil in
#     sedpy.observate.list_available_filters() if ('jwst_f' in fil) & (fil.endswith('m'))])
mbfils = np.array(sedpy.observate.load_filters(['jwst_'+f for f in mb_filters]))

# figure out which band has lines -- Pa-beta and Ha
linewl_z_pab = 1.282 * (sample['z']+1)
linewl_z_ha = .6563 * (sample['z']+1)
lineinfil_pab = np.zeros((len(linewl_z_pab)), dtype='<U10')
lineinfil_ha = np.zeros((len(linewl_z_pab)), dtype='<U10')
for ii in range(len(linewl_z_pab)):
    print(sample[ii]['ID'])
    try:
        lineinfil_pab[ii] = [key.lower() for key, value in filter_wavelengths.items() if value[0]<=linewl_z_pab[ii]<=value[1]][0]
    except IndexError:
        print('no PaB')
    
    try:
        lineinfil_ha[ii] = [key.lower() for key, value in filter_wavelengths.items() if value[0]<=linewl_z_ha[ii]<=value[1]][0]   
    except IndexError:
        print('no Ha')         
    
# select just objects that have lines in both filters
sel = np.array([len(i)>0 for i in lineinfil_pab]) & np.array([len(i)>0 for i in lineinfil_ha]) & (lineinfil_ha != 'f140m') &  (lineinfil_pab != 'f480m')
sample = sample[sel]
lineinfil_pab = lineinfil_pab[sel]
lineinfil_ha = lineinfil_ha[sel]

# theoretical scalings (to Hb, from naveen's paper)
ha_factor = 2.79
pab_factor = 0.155


bluecol = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
for ii, msa_id in enumerate(sample['ID'].astype('int')):
    with fits.open('/Volumes/DarkPhoenix/Surveys/UNCOVER/spectra/uncover-v2_prism-clear_2561_'+str(msa_id)+'.spec.fits') as hdu:
        spec_wl = hdu[1].data['wave']
        spec_fl = hdu[1].data['flux']
        
    # get cutouts
    if msa_id not in mbphot['id_msa']: 
        print('MSA ID '+str(msa_id)+' MISSING!')
        continue
    cuts = get_all_cutouts(mbphot[mbphot['id_msa']==msa_id]['ra'], mbphot[mbphot['id_msa']==msa_id]['dec'], dt=30)
    
    # get line, red/blue cont for PaB
    pab_map = cuts[filters==lineinfil_pab[ii]][0]
    pab_bluecont = cuts[filters== mb_filters[np.where(mb_filters==lineinfil_pab[ii])[0][0]-1]][0]
    pab_redcont = cuts[filters== mb_filters[np.where(mb_filters==lineinfil_pab[ii])[0][0]+1]][0]
    pab_contmap = np.mean(np.array([pab_redcont, pab_bluecont]), axis=0)
    pab_linemap = pab_map - pab_contmap
    
    # get sedpy objects for these filters
    pab_sed_linefil = mbfils[mb_filters==lineinfil_pab[ii]][0]
    pab_sed_bluefil = mbfils[np.where(mb_filters==lineinfil_pab[ii])[0][0]-1]
    pab_sed_redfil = mbfils[np.where(mb_filters==lineinfil_pab[ii])[0][0]+1]
    
    # get line, red/blue cont for Ha+[NII]
    ha_map = cuts[filters==lineinfil_ha[ii]][0]
    ha_bluecont = cuts[filters== mb_filters[np.where(mb_filters==lineinfil_ha[ii])[0][0]-1]][0]
    ha_redcont = cuts[filters== mb_filters[np.where(mb_filters==lineinfil_ha[ii])[0][0]+1]][0]
    ha_contmap = np.mean(np.array([ha_bluecont, ha_redcont]), axis=0)
    ha_linemap = ha_map - ha_redcont
    
    # get sedpy objects for Ha filters
    ha_sed_linefil = mbfils[mb_filters==lineinfil_ha[ii]][0]
    ha_sed_bluefil = mbfils[np.where(mb_filters==lineinfil_ha[ii])[0][0]-1]
    ha_sed_redfil = mbfils[np.where(mb_filters==lineinfil_ha[ii])[0][0]+1]
    

   
    # initialize figure
    f, ax = plt.subplots(2,4, figsize=(10.5,6), gridspec_kw={'width_ratios':[1.5, 1, 1, 1]}) 
    f.subplots_adjust(wspace=0.1)   
    
    # first axis: plot the spectrum + transmission filters (Ha)
    ax[0,0].step(spec_wl, spec_fl/np.max(spec_fl[(spec_wl>3) & (spec_wl<4.5)]), color='black', lw=0.75)
    ax[0,0].plot(ha_sed_bluefil.wavelength/1e4, ha_sed_bluefil.transmission, color=bluecol, alpha=0.75)
    ax[0,0].plot(ha_sed_linefil.wavelength/1e4, ha_sed_linefil.transmission, color='forestgreen', alpha=0.75)
    ax[0,0].plot(ha_sed_redfil.wavelength/1e4, ha_sed_redfil.transmission, color='firebrick', alpha=0.75)
    ax[0,0].set_xlim((1,5))
    ax[0,0].set_ylim((0,1.1))
    ax[0,0].set_xlabel(r'$\lambda_{\rm{obs}}\ (\mu\rm{m})$')
    ax[0,0].set_ylabel(r'$f_{\rm{\nu}}$ (norm)')
    ax[0,0].set_yticklabels([])
    ax[0,0].text(0.05, 0.95, 'ID '+str(msa_id)+'\n'+
        r'$z_{\rm{spec}}=$'+'{:.2f}'.format(sample[ii]['z']), transform=ax[0,0].transAxes, fontsize=14, va='top' )
        
    # now, plot cutouts!
    vmin = np.percentile(pab_linemap/pab_factor, 10)
    vmax = np.percentile(pab_linemap/pab_factor, 99)
    
    if vmax<0: vmax = np.max(ha_linemap)*.9
    # first cutout: color of F335M / F360M / F410M
    ax[0,1].imshow(make_linear_rgb(ha_bluecont, ha_map, ha_redcont, minimum=1e-3, maximum=0.75*np.max(np.array([ha_bluecont, ha_map, ha_redcont]))), origin='lower')
    # next: continuum = median of F335 and F410M
    cmap = sns.cubehelix_palette(as_cmap=True, reverse=False, dark=0, light=1) #'rocket_r'
    ax[0,2].imshow(ha_contmap, origin='lower', vmin=np.percentile(ha_contmap,10), vmax=np.percentile(ha_contmap,99), cmap=cmap)
    ax[0,2].text(0.05, 0.975, 'Ha continuum', fontsize=14, transform=ax[0,2].transAxes, va='top')
    # last: line = line MB - continuum
    ax[0,3].imshow(ha_linemap/ha_factor, origin='lower', vmin=vmin, vmax=vmax/3, cmap=cmap)
    ax[0,3].text(0.05, 0.975, 'Ha line', fontsize=14, transform=ax[0,3].transAxes, va='top')
    # put a 0.5" scalebar
    ax[0,1].plot([5, 5+0.5 / 0.04], [5,5], color='white')
    ax[0,2].plot([5, 5+0.5 / 0.04], [5,5], color='black')
    ax[0,3].plot([5, 5+0.5 / 0.04], [5,5], color='black')

    ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
    ax[0,2].set_xticks([]); ax[0,2].set_yticks([])
    ax[0,3].set_xticks([]); ax[0,3].set_yticks([])
    
    
    ####### same thing but PaB
    # first axis: plot the spectrum + transmission filters (Ha)
    ax[1,0].step(spec_wl, spec_fl/np.max(spec_fl[(spec_wl>3) & (spec_wl<4.5)]), color='black', lw=0.75)
    ax[1,0].plot(pab_sed_bluefil.wavelength/1e4, pab_sed_bluefil.transmission, color=bluecol, alpha=0.75)
    ax[1,0].plot(pab_sed_linefil.wavelength/1e4, pab_sed_linefil.transmission, color='forestgreen', alpha=0.75)
    ax[1,0].plot(pab_sed_redfil.wavelength/1e4, pab_sed_redfil.transmission, color='firebrick', alpha=0.75)
    ax[1,0].set_xlim((1,5))
    ax[1,0].set_ylim((0,1.1))
    ax[1,0].set_xlabel(r'$\lambda_{\rm{obs}}\ (\mu\rm{m})$')
    ax[1,0].set_ylabel(r'$f_{\rm{\nu}}$ (norm)')
    ax[1,0].set_yticklabels([])
        
    # now, plot cutouts!
    # first cutout: color of F335M / F360M / F410M
    ax[1,1].imshow(make_linear_rgb(pab_bluecont, pab_map, pab_redcont, minimum=1e-3, maximum=0.75*np.max(np.array([pab_bluecont, pab_map, pab_redcont]))), origin='lower')
    # next: continuum = median of F335 and F410M
    cmap = sns.cubehelix_palette(as_cmap=True, reverse=False, dark=0, light=1) #'rocket_r'
    ax[1,2].imshow(pab_contmap, origin='lower', vmin=np.percentile(pab_contmap,10), vmax=np.percentile(pab_contmap,99), cmap=cmap)
    ax[1,2].text(0.05, 0.975, 'PaB continuum', fontsize=14, transform=ax[1,2].transAxes, va='top')
    # last: line = line MB - continuum
    ax[1,3].imshow(pab_linemap/pab_factor, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1,3].text(0.05, 0.975, 'PaB line', fontsize=14, transform=ax[1,3].transAxes, va='top')
    # put a 0.5" scalebar
    ax[1,1].plot([5, 5+0.5 / 0.04], [5,5], color='white')
    ax[1,2].plot([5, 5+0.5 / 0.04], [5,5], color='black')
    ax[1,3].plot([5, 5+0.5 / 0.04], [5,5], color='black')

    ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
    ax[1,2].set_xticks([]); ax[1,2].set_yticks([])
    ax[1,3].set_xticks([]); ax[1,3].set_yticks([])
        
    for axi in range(1,4):
        ax[0,axi].scatter(ha_linemap.shape[0]/2-1, ha_linemap.shape[0]/2, marker='x', color='0.3')
        ax[1,axi].scatter(ha_linemap.shape[0]/2-1, ha_linemap.shape[0]/2, marker='x', color='0.3')

    f.tight_layout()
    f.subplots_adjust(wspace=0)
    f.savefig('plots/'+str(msa_id)+'.pdf', bbox_inches='tight')
    plt.close()
    




        		
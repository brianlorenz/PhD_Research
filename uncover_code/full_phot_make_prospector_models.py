import numpy as np
from astropy.io import fits
import sys
# sys.path.insert(0, '/Users/wren/Projects/prospector_catalog/')
from prospect.models.sedmodel import PolySedModel
from prospect.models.sedmodel import PolySpecModel
# sys.path.insert(0, '/Users/brianlorenz/prospector_catalog-main/')
# sys.path.insert(0, '/Users/brianlorenz/fsps')
from params_prosp_fsps import params_fsps_phisfh, build_sps_fsps
import utils as ut_cwd
from astropy.table import Table
import sedpy
import copy
np.infty = np.inf # stupid prospector hack because my numpy is too high a version
import pandas as pd

npz_loc = '/Users/brianlorenz/uncover/Data/prospector_spectra_no_emission/spectra.npz'
prospector_abs_spec_folder = '/Users/brianlorenz/uncover/Data/prospector_spectra_no_emission/specs/'

def generate_prospector_models(id_dr3_list):
    # load catalogs
    catalog_file = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
    with fits.open(catalog_file) as hdu:
        mbphot = hdu[1].data  
    cat = Table.read(catalog_file)    
    filter_dict = ut_cwd.filter_dictionary(mb=True, alma=False)
    filts = list(filter_dict.keys())
    filternames = list(filter_dict.values())

    # load in prospector catalog
    with fits.open('/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits') as hdu:
        prospect = hdu[1].data    
        
    # generate the model    
    params, fit_order = params_fsps_phisfh()       
    sps = build_sps_fsps()

    # and chains
    chain_1 = np.load('/Users/BrianLorenz/uncover/Catalogs/chains_v5.3.0_LW_SUPER_spsv1.0.npz', allow_pickle=True)    
    chain_2 = np.load('/Users/BrianLorenz/uncover/Catalogs/chains_sfrr_v5.3.0_LW_SUPER_spsv1.0.npz', allow_pickle=True)    

    # put together the MAP theta for our galaxies
    # this is annoying because the chains are split over two objects and not actually named the same thing
    # per bingjie on slack, the MAP (or max lnL) value is the last entry in the resampled chain
    spectra = {}
    model = PolySpecModel(params)
    model_theta_labels = model.theta_labels()
    def load_obs(idx=None, err_floor=0.05, **extras):
        '''
        idx: obj idx in the catalog
        copied from bingjie github so that we don't have to have mu maps downloaded
        '''

        from prospect.utils.obsutils import fix_obs

        flux = ut_cwd.get_fnu_maggies(idx, cat, filts)
        unc = ut_cwd.get_enu_maggies(idx, cat, filts)

        obs = {}
        obs["filters"] = sedpy.observate.load_filters(filternames)
        obs["wave_effective"] = np.array([f.wave_effective for f in obs["filters"]])
        obs["maggies"] = flux
        obs["maggies_unc"] = unc
        # define photometric mask
        # mask out fluxes with negative errors, and high-confidence negative flux
        phot_mask = (unc > 0) & (np.isfinite(flux))
        _mask = np.ones_like(unc, dtype=bool)
        for k in range(len(flux)):
            if unc[k] > 0:
                if flux[k] < 0 and flux[k] + 5*unc[k] < 0:
                    _mask[k] = False
        phot_mask &= _mask
        obs['phot_mask'] = phot_mask
        # impose minimum error floor
        obs['maggies_unc'] = np.clip(obs['maggies_unc'], a_min=obs['maggies']*err_floor, a_max=None)

        obs["wavelength"] = None
        obs["spectrum"] = None
        obs['unc'] = None
        obs['mask'] = None

        # other useful info
        obs['objid'] = cat['id'][idx]
        obs['catalog'] = catalog_file

        obs = fix_obs(obs)

        return obs

    for galid in id_dr3_list:
        # get our galaxy (this is sloooooow)
        print('galaxy '+str(galid))
        chain1 = chain_1['chains'][chain_1['objid'] == galid]
        chain2 = chain_2['chains'][chain_2['objid'] == galid]
        obs = load_obs(idx=np.where(mbphot['id']==galid)[0][0])
        spectra[galid] = np.zeros((len(sps.wavelengths),3))
        spectra[galid][:,0] = sps.wavelengths
        
        # concatenate into the right MAP theta
        map_theta = np.concatenate((chain2[0,-1,:2], chain1[0,-1,chain_1['theta_labels']=='logzsol'], chain2[0,-1,2:], 
            np.array([chain1[0,-1,chain_1['theta_labels']==l] for l in model_theta_labels[9:]]).flatten()))
        
        # get standard model
        print('getting nebular model') 
        model.params['add_neb_emission'][0] = True
        model.params['add_neb_continuum'][0] = True
        assert model.params['add_neb_emission'][0] == True
        spec, phot, mfrac = spec, phot, mfrac = model.predict(map_theta, sps=sps, obs=obs)  
        spec, phot, mfrac = spec, phot, mfrac = model.predict(map_theta, sps=sps, obs=obs)  
        spectra[galid][:,1] = copy.deepcopy(spec) # model.predict is finnicky, gotta save now otherwise calling it again overwrites spec

        
        # get model without nebular emission
        print('getting continuum model')
        model.params['add_neb_emission'][0] = False
        model.params['add_neb_continuum'][0] = False    
        spec_noneb, phot_noneb, mfrac = spec, phot, mfrac = model.predict(map_theta, sps=sps, obs=obs)
        spectra[galid][:,2] = copy.deepcopy(spec_noneb)
    np.savez(npz_loc, spectra=spectra) 

    for id_dr3 in id_dr3_list:
        absorp_npz_to_csv(id_dr3)

def absorp_npz_to_csv(id_dr3):
    np_df = np.load(npz_loc, allow_pickle=True)
    np_dict = np_df['spectra'].item()
    rest_wave = np_dict[id_dr3][:,0]
    full_model = np_dict[id_dr3][:,1]
    absorp_model = np_dict[id_dr3][:,2]
    cont_df = pd.DataFrame(zip(rest_wave, full_model, absorp_model), columns=['rest_wave', 'rest_full_model', 'rest_absorp_model_maggies'])
    # Not actually rest flux - take a look at this
    cont_df['rest_absorp_model_jy'] =  cont_df['rest_absorp_model_maggies'] * 3631
    cont_df['rest_absorp_model_10njy'] =  cont_df['rest_absorp_model_jy'] / 1e8
    cont_df['rest_full_model_jy'] =  cont_df['rest_full_model'] * 3631
    c = 299792458 # m/s
    cont_df['rest_absorp_model_erg_aa'] = cont_df['rest_absorp_model_jy'] * (1e-23*1e10*c / (cont_df['rest_wave']**2))
    
    cont_df.to_csv(f'{prospector_abs_spec_folder}{id_dr3}_prospector_no_neb.csv', index=False)
    return        
    

if __name__ == "__main__":
    # id_dr3_list = [20686, 22045]
    # generate_prospector_models(id_dr3_list)

    # absorp_npz_to_csv(30052)

    # project_1_ids = np.array([30052, 30804, 31608, 37182, 37776, 44283, 46339, 47771, 49023, 52140, 52257, 54625, 60579, 62937])
    # for id in project_1_ids:
    #     absorp_npz_to_csv(id)

    full_gals_list = [17757, 17758, 30052, 30351, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780, 13130, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]
    generate_prospector_models(full_gals_list)
    # project_1_id_list = [30052, 30804, 31608, 37182, 37776, 44283, 46339, 47771, 49023, 52140, 52257, 54625, 60579, 62937]
    # generate_prospector_models(project_1_id_list)

    
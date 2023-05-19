import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from mosdef_obj_data_funcs import read_sed, read_mock_sed
from cross_correlate import get_cross_cor
from composite_sed import vis_composite_sed

demo_folder = imd.cluster_dir + '/clustering_demo'

def show_process(field1, v4id1, field2, v4id2):
    sed1 = read_sed(field1, v4id1, norm=False)
    sed2 = read_sed(field2, v4id2, norm=False)

    ### OBSERVED FRAME ###
    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(sed1['peak_wavelength'], sed1['f_lambda']*0.5, marker='o', color='orange', ls='None')
    ax.plot(sed2['peak_wavelength'], sed2['f_lambda'], marker='o', color='blue', ls='None')
    ax.set_xscale('log')
    ax.set_xlabel('Observed Wavelength ($\AA$)')
    ax.set_ylabel('F_$\lambda$')
    fig.savefig(demo_folder + '/observed_frame.pdf', bbox_inches='tight')
    plt.close('all')

    ### REST FRAME ###
    sed1['rest_wavelength'] = sed1['peak_wavelength'] / (1+sed1['Z_MOSFIRE'])
    sed1['rest_flux'] = sed1['f_lambda'] * (1+sed1['Z_MOSFIRE'])

    sed2['rest_wavelength'] = sed2['peak_wavelength'] / (1+sed2['Z_MOSFIRE'])
    sed2['rest_flux'] = sed2['f_lambda'] * (1+sed2['Z_MOSFIRE'])
    
    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(sed1['rest_wavelength'], sed1['rest_flux']*0.5, marker='o', color='orange', ls='None')
    ax.plot(sed2['rest_wavelength'], sed2['rest_flux'], marker='o', color='blue', ls='None')
    ax.set_xscale('log')
    ax.set_xlabel('Rest Wavelength ($\AA$)')
    ax.set_ylabel('F_$\lambda$')
    ax.set_ylim(0e-18, 4e-18)
    fig.savefig(demo_folder + '/rest_frame.pdf', bbox_inches='tight')
    plt.close('all')

    ### MOCK SED ###
    mock_sed1 = read_mock_sed(field1, v4id1)
    mock_sed2 = read_mock_sed(field2, v4id2)
    a12, b12 = get_cross_cor(mock_sed1, mock_sed2)
    print(f'scale: {a12}')
    print(f'corr: {b12}')
    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(sed1['rest_wavelength'], sed1['rest_flux'], marker='o', color='grey', ls='None', alpha=0)
    ax.plot(sed2['rest_wavelength'], sed2['rest_flux'], marker='o', color='grey', ls='None', alpha=0)
    mock_sed1 = mock_sed1[mock_sed1['f_lambda'] > -80]
    mock_sed2 = mock_sed2[mock_sed2['f_lambda'] > -80]
    ax.plot(mock_sed1['rest_wavelength'], 0.5*mock_sed1['f_lambda']* (1+sed1['Z_MOSFIRE'].iloc[0]), marker='o', color='orange', ls='None')
    ax.plot(mock_sed2['rest_wavelength'], mock_sed2['f_lambda']* (1+sed2['Z_MOSFIRE'].iloc[0]), marker='o', color='blue', ls='None')
    ax.set_xscale('log')
    ax.set_xlabel('Rest Wavelength ($\AA$)')
    ax.set_ylabel('F_$\lambda$')
    ax.set_ylim(0e-18, 4e-18)
    fig.savefig(demo_folder + '/mock_sed.pdf', bbox_inches='tight')
    plt.close('all')

    ### Scaled ###
    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(sed1['rest_wavelength'], sed1['rest_flux']*(1/a12), marker='o', color='orange', ls='None')
    ax.plot(sed2['rest_wavelength'], sed2['rest_flux'], marker='o', color='blue', ls='None')
    ax.set_xscale('log')
    ax.set_xlabel('Rest Wavelength ($\AA$)')
    ax.set_ylabel('F_$\lambda$')
    ax.set_ylim(0e-18, 4e-18)
    fig.savefig(demo_folder + '/rest_frame_scaled.pdf', bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(figsize=(6,6))
    vis_composite_sed(0, groupID=0, axis_obj=ax, run_filters=False, grey_points=True)
    fig.savefig(demo_folder + '/composite.pdf', bbox_inches='tight')


show_process('AEGIS', 19375, 'COSMOS', 11968)


def plot_all_composites(n_clusters):
    for groupID in range(n_clusters):
        fig, ax = plt.subplots(figsize=(6,6))
        vis_composite_sed(0, groupID=groupID, axis_obj=ax, run_filters=False, grey_points=True)
        fig.savefig(demo_folder + f'/composite_{groupID}.pdf', bbox_inches='tight')


# plot_all_composites(23)
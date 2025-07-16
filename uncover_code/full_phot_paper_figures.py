from full_phot_sample_select_sfr_mass import plot_paper_sample_select_sfr_mass
from full_phot_property_vs_dust import plot_paper_dust_vs_prop
from full_phot_read_data import read_phot_df

if __name__ == '__main__':
    # plot_paper_sample_select_sfr_mass()
    # plot_paper_sample_select_sfr_mass(show_hexes=True)
    plot_paper_sample_select_sfr_mass(show_hexes=True, mass_cut=8)

    props = ['mass', 'sfr', 'axisratio_f444w', 'axisratio_f150w', 'axisratio_halpha']
    color_vars = ['snr', 'redshift']
    phot_df = read_phot_df()
    for prop in props:
        for color_var in color_vars:
            plot_paper_dust_vs_prop(prop=prop, color_var=color_var, phot_df=phot_df)
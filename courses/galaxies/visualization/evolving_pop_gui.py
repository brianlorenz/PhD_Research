# GUI to visualize how a galaxy spectrum evolves over time
'''


'''


import fsps
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
from generate_ssp import get_filename, ages_allowed


os.environ["SPS_HOME"] = "/Users/galaxies-air/SPS_Conroy/fsps/"

loc = '/Users/galaxies-air/Desktop/Galaxies/visualization/'

ssp_folder = '/Users/galaxies-air/Desktop/Galaxies/visualization/ssps/'


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class StellarPop:
    def __init__(self, age=0):
        # Read the initial SSP in here
        # Main variable to contain running total of ssps. Fromat as (birthtime, metallicity, dust)
        self.ssps = self.create_ssps_var()
        self.frozen = -1
        self.gal_age = 0.0  # Current age of the galaxy
        self.gal_dust = 0.0  # Current dust content of the galaxy
        self.zoom_limits = [3000, 7000]  # Region of spectrum to zoom in on
        # File that stores wavelength for sps
        self.wavelength = pd.read_csv(ssp_folder+'wavelength.df')
        self.frequency = (3*10**18)/self.wavelength
        initial_ssp = self.create_sp()
        # Running total of ssps, will iterate over this for every observable
        self.ssps = self.ssps.append(initial_ssp, ignore_index=True)
        self.create_plot()
        self.initialize_buttons()
        self.update_plot(self.ssps)

    def create_ssps_var(self):
        """Creates the self.ssps variable that is used throughout the code

        Parameters:


        Returns:
        ssps (pd.Dataframe): Dataframe setup in the proper format
        """
        return pd.DataFrame(columns=['Birthtime', 'Metallicity', 'Dust'])

    def create_sp(self, metallicity=0.0, birthtime=0.0):
        """Creates a stellar population dict that can easily be appended to self.ssps dataframe

        Parameters:
        metallicity (float): log(Z) in solar units (so 0.0 is solar metallicity)
        age (float): how long after galaxy formation to birth this stellar population


        Returns:
        sp (dict): dict containing birthtime, metallicity
        """
        sp = {'Birthtime': birthtime, 'Metallicity': metallicity}
        return sp

    def read_sp(self, metallicity=0.0, age=1.0, dust=0.0):
        """Reads the stellar population model with the given parameters.

        Parameters:
        metallicity (float): log(Z) in solar units (so 0.0 is solar metallicity)
        age (float): how long after galaxy formation to birth this stellar population
        dust (float): how much dust there is

        Returns:
        pd.Dataframe: Dataframe containing spectrum and fraction of stars left at that age, Z, dust

        # MUST BE FORMATTED AS 0.0, 1.0, 0.0. Floats need decimals
        """

        filename = get_filename(metallicity=metallicity, dust=dust, age=age)
        sp_df = pd.read_csv(ssp_folder+filename)
        print(f'Read {filename}')
        return sp_df

    def create_plot(self):
        # Sets up the figure for first time use. Called from __init__
        '''

        '''
        self.fig = plt.figure(figsize=(10, 8))
        # Axis where the spectrum is
        self.ax_spec = self.fig.add_axes([0.1, 0.63, 0.8, 0.35])
        self.ax_zoom = self.fig.add_axes([0.1, 0.3, 0.8, 0.25])

        self.nans_arr = np.ones(len(self.wavelength))
        self.nans_arr[:] = np.nan

        self.spectrum, = self.ax_spec.plot(
            self.wavelength, np.ones(
                len(self.wavelength)), label='Spectrum', color='black')
        self.spectrum_freeze, = self.ax_spec.plot(
            self.wavelength, self.nans_arr, label='Spectrum', color='red', alpha=0.5)
        self.zoom_spectrum, = self.ax_zoom.plot(self.wavelength, np.ones(
            len(self.wavelength)), label='Spectrum', color='black')
        self.zoom_spectrum_freeze, = self.ax_zoom.plot(
            self.wavelength, self.nans_arr, label='Spectrum', color='red', alpha=0.5)

        self.ax_spec.set_xlabel('Wavelength ($\\AA$)')
        self.ax_spec.set_ylabel('Intensity (L$_\odot$)')
        self.ax_zoom.set_xlabel('Wavelength ($\\AA$)')
        self.ax_zoom.set_ylabel('Intensity (L$_\odot$)')

        self.ax_spec.set_xscale('log')
        self.ax_spec.set_yscale('log')
        self.ax_zoom.set_xscale('log')
        self.ax_zoom.set_yscale('log')

        plt.show()

    def get_sp_files(self, ssps, gal_age, gal_dust):
        sp_files = []
        for i in range(len(ssps)):
            metallicity = ssps.iloc[i]['Metallicity']
            birthtime = ssps.iloc[i]['Birthtime']
            # Comptue the age of the current population as galaxy_age - population_birthtime
            age = gal_age - birthtime
            if age >= 0:
                age = find_nearest(ages_allowed, age)
                sp_file = self.read_sp(
                    metallicity=metallicity, age=age, dust=self.gal_dust)
                sp_files.append(sp_file)
        return sp_files

    def update_plot(self, ssps, gal_age=0.0, gal_dust=0.0):
        """Updates the figure after a change to the ssps

        Parameters:
        ssps (pd.Dataframe): self.ssps variable that gets passed around. Stores all ssps and their ages

        Returns:

        """
        sp_files = self.get_sp_files(ssps, gal_age, gal_dust)
        print(len(sp_files))
        self.spectra = [sp_files[i]['Spectrum'] for i in range(len(sp_files))]

        if self.frozen == 1:
            self.spectrum_freeze.set_ydata(self.total_spectrum)
            self.zoom_spectrum_freeze.set_ydata(self.total_spectrum)
            self.ax_spec.text(10**7, 100, f'Age: {np.round(self.gal_age,2)} Gyr', color='red', alpha=0.5)
            self.ax_spec.text(10**7, 10, f'log Z: {ssps.iloc[0]["Metallicity"]} Z$_\odot$', color='red', alpha=0.5)
            self.ax_spec.text(10**7, 1, f'Dust: {self.gal_dust}', color='red', alpha=0.5)
            self.frozen_spec = self.spectrum_freeze.get_ydata()
            self.frozen = 0

        if self.frozen == -1:
            self.spectrum_freeze.set_ydata(self.nans_arr)
            self.zoom_spectrum_freeze.set_ydata(self.nans_arr)
            for txt in self.ax_spec.texts:
                txt.set_visible(False)

        # Multiply by nu since units of spectrum are Lsun/nu
        self.total_spectrum = np.multiply(
            np.transpose(np.array(self.frequency)), np.sum(self.spectra, axis=0))

        self.spectrum.set_ydata(self.total_spectrum)

        self.zoom_spectrum.set_ydata(self.total_spectrum)

        #
        self.ax_zoom.set_xlim(self.zoom_limits)

        # self.set_axis_limits(self.ax_spec)
        self.ax_spec.set_ylim([10**-12, 1000])
        wavelength = np.array(self.wavelength['Wavelength'].to_list())

        # Find the region that we are currently zoomed over
        idx = np.logical_and(
            wavelength > self.zoom_limits[0], wavelength < self.zoom_limits[-1])
        zoom_lower_lim = np.max([
            np.min(self.total_spectrum[0][idx]) * 0.5, 10**-12])
        zoom_upper_lim = np.max(self.total_spectrum[0][idx])*2
        if self.frozen == 0:
            freeze_lower_lim = np.min(self.frozen_spec[0][idx])
            freeze_upper_lim = np.max(self.frozen_spec[0][idx])
            zoom_lower_lim = np.min([freeze_lower_lim, zoom_lower_lim])
            zoom_upper_lim = np.max([freeze_upper_lim, zoom_upper_lim])
        self.ax_zoom.set_ylim([zoom_lower_lim, zoom_upper_lim])
        # self.set_axis_limits(self.ax_zoom)
        plt.draw()

    def initialize_buttons(self):
        """Gets the buttons and sliders ready

        Parameters:

        Returns:

        """

        # Sliders for age, metal, dust
        slidercolor = 'dodgerblue'
        ax_age = self.fig.add_axes([0.1, 0.18, 0.8, 0.03])
        self.age_slider = Slider(
            ax_age, 'Age (Gyr)', 0.0, 15.0, valinit=0, valstep=0.01, color=slidercolor)
        self.age_slider.on_changed(self.set_age)

        ax_metal = self.fig.add_axes([0.1, 0.13, 0.8, 0.03])
        self.metal_slider = Slider(ax_metal, 'log Z (Z$_\odot$)', -1.0, 1.0,
                                   valinit=0, valstep=0.25, color=slidercolor)
        self.metal_slider.on_changed(self.set_metal)

        ax_dust = self.fig.add_axes([0.1, 0.08, 0.8, 0.03])
        self.dust_slider = Slider(ax_dust, 'Dust', 0.0, 1.0,
                                  valinit=0, valstep=0.25, color=slidercolor)
        self.dust_slider.on_changed(self.set_dust)

        # Button to add spectrum
        ax_adsp = self.fig.add_axes([0.1, 0.02, 0.15, 0.03])
        self.ax_adsp = Button(ax_adsp, 'Starburst')
        self.ax_adsp.on_clicked(self.button_add_sp)

        # Button to freeze the current spectrum
        ax_freeze = self.fig.add_axes([0.3166, 0.02, 0.15, 0.03])
        self.ax_freeze = Button(ax_freeze, 'Freeze')
        self.ax_freeze.on_clicked(self.button_freeze)

        # Button to play a movie
        ax_movie = self.fig.add_axes([0.5333, 0.02, 0.15, 0.03])
        self.ax_movie = Button(ax_movie, 'Play Movie')
        self.ax_movie.on_clicked(self.button_play_movie)

        # Button to reset all sp
        ax_reset = self.fig.add_axes([0.75, 0.02, 0.15, 0.03])
        self.ax_reset = Button(ax_reset, 'Reset')
        self.ax_reset.on_clicked(self.button_reset)

        # Slider for zoom region
        self.zoom_span = SpanSelector(self.ax_spec, self.zoomslider, 'horizontal',
                                      useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

    def set_age(self, age):
        """Sets the age of the galaxy to the given value, then updates the plot

        Parameters:
        age (float): age of the galaxy

        Returns:

        """
        self.gal_age = age
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def set_metal(self, metal):
        """Sets the metallicity of the galaxy to the given value, then updates the plot

        Parameters:
        metal (float): metallicity of the galaxy

        Returns:

        """
        self.ssps['Metallicity'] = metal
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def add_sp(self, birthtime=1.0, metallicity=0.0):
        """Adds a starburst at the current time, with its own birthtime and metallicity

        Parameters:
        birthtime (float): time that the sp was formed (Gyr)
        metallicity (float): Z in log solar

        Returns:
        None: Updates the self.ssps dataframe with new row appended

        """
        sp = self.create_sp(birthtime=birthtime, metallicity=metallicity)
        self.ssps = self.ssps.append(sp, ignore_index=True)
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def set_dust(self, dust):
        """Sets the dust of the galaxy to the given value, then updates the plot

        Parameters:
        dust (float): age of the galaxy

        Returns:

        """
        self.gal_dust = dust
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def button_add_sp(self, event):
        print(f'Starburst at {self.gal_age} Gyr!')
        self.add_sp(birthtime=self.gal_age)

    def button_play_movie(self, event):
        print(f'Playing Movie')
        moviespeed = 0.1
        for i in np.arange(0, 15+moviespeed, moviespeed):
            self.gal_age = i
            self.update_plot(self.ssps, gal_age=self.gal_age,
                             gal_dust=self.gal_dust)
            self.age_slider.set_val(i)
            plt.pause(0.1)

    def button_reset(self, event):
        print(f'Resetting')
        self.ssps = self.create_ssps_var()
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def button_freeze(self, event):
        print(f'Freezing')
        self.frozen = - self.frozen
        if self.frozen == 0:
            self.frozen = -1
        if self.frozen == -1:
            self.ax_freeze.label.set_text('Freeze')
        else:
            self.ax_freeze.label.set_text('Unfreeze')
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)

    def zoomslider(self, xmin, xmax):
        self.zoom_limits = [xmin, xmax]
        self.update_plot(self.ssps, gal_age=self.gal_age,
                         gal_dust=self.gal_dust)


stellar_pop = StellarPop()

# Simulate distribution of viewing angles

import os
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from read_data import mosdef_df
from astropy.io import fits
from mpl_toolkits import mplot3d
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse
from scipy import spatial






def plot_ar_distribution():
    ## Figure 1 - thin disk - formula is just Rsin(theta) / R for random thetas, so we get

    thetas = np.random.rand(10000) * (np.pi/2)
    phis = np.random.rand(10000) * (np.pi/2)
    axis_ratios = np.sin(thetas)/np.cos(thetas)

    fig, ax = plt.subplots(figsize=(8,8))

    ax.hist(axis_ratios, 20, color='black')

    ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(-1.05, 1.05)

    plt.show()


    ## Figure 2 -  disk with radius R - 

    thetas = np.random.rand(10000) * (np.pi/2)
    phis = np.random.rand(10000) * (np.pi/2)
    axis_ratios = np.sin(thetas)/np.cos(thetas)

    fig, ax = plt.subplots(figsize=(8,8))

    ax.hist(axis_ratios, 20, color='black')

    ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(-1.05, 1.05)

    plt.show()




def make_shape(n_points = 1000, height = 1):
    radius = 1

    
    def generate_point(radius, height):
        '''Generates a point on a cylinder'''
        # Generate points on the outside of a cylinder
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(0, height)
        r =  radius
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        point = np.array([x, y, z])
        return point

    def random_point_ellipsoid(a,b,c):
        '''a, b, c are the axes'''
        u = np.random.rand()
        v = np.random.rand()
        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        rx = a * sinPhi * cosTheta
        ry = b * sinPhi * sinTheta
        rz = c * cosPhi
        point = np.array([rx, ry, rz])
        return point

    points = []
    for i in range(n_points):
        ellipticity = np.random.rand()*0.2+0.8
        points.append(random_point_ellipsoid(1,ellipticity,height))

    return points



def rotate_cylinder(points, only_rot_z = False):
    '''Spins the cylinder around the x axes (don't need to do y or z)
    
    Parameters:
    only_rot_z (boolean): If True, don't rotate over the x-y planes. Used for testing with known ellipses
    '''
    theta = np.random.uniform(0, 2*np.pi)
    # phi = np.random.uniform(0, 2*np.pi)
    # rho = np.random.uniform(0, 2*np.pi)
    rot_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    # rot_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    # rot_z = np.array([[np.cos(rho), -np.sin(rho), 0], [np.sin(rho), np.cos(rho), 0], [0, 0, 1]])
    if only_rot_z==False:
        points = [np.matmul(rot_x, point) for point in points]
        # points = [np.matmul(rot_y, point) for point in points]
    # points = [np.matmul(rot_z, point) for point in points]
    return points


def plot_points(points, show_plots=False):
    '''Takes the 3d array points and plots it in 3d space'''
    xdata = [point[0] for point in points]
    ydata = [point[1] for point in points]
    zdata = [point[2] for point in points]
    
    def make_3d_plot():
        '''Shows the 3d plot of the object'''
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata, ydata, zdata)

        ax.axes.set_xlim3d(left=-2, right=2) 
        ax.axes.set_ylim3d(bottom=-2, top=2) 
        ax.axes.set_zlim3d(bottom=-2, top=2) 

        plt.show()
        plt.close('all')
    if show_plots==True:
        make_3d_plot()


    ## Fit the ellipse https://github.com/bdhammel/least-squares-ellipse-fitting
    X = np.array(list(zip(xdata, ydata)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()


    def make_2d_plot():
        '''Shows the 2d plot of the object'''
        fig, ax = plt.subplots()
        ax.scatter(xdata, ydata)

        ellipse = Ellipse(
            xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax.add_patch(ellipse)


        ax.set_xlim(-2, 2) 
        ax.set_ylim(-2, 2) 

        plt.show()
        plt.close('all')



    def compute_axis_ratio():
        '''Calculates the axis ratio from the ellipse'''

        if width > height:
            axis_ratio = height/width
        else: 
            axis_ratio = width/height

        return axis_ratio

    axis_ratio = compute_axis_ratio()
    if show_plots==True:
        make_2d_plot()
        print(f'Axis Ratio: {axis_ratio}')
    return axis_ratio, center, width, height, phi

def remove_ellipse_point(point, width, height, phi):
    '''Returns True if the point is in the ellipse, otherwise false. Assumes ellipse is centered at (0,0)
    '''
    x = point[0]
    y = point[1]
    in_ellipse = ((x * np.cos(phi) + y*np.sin(phi))**2 / (width**2) + (x * np.sin(phi) - y*np.cos(phi))**2 / (height**2)) < 1
    return in_ellipse
    

def iterate_remove_points(points, n_iter, show_plots=False):
    '''Function that will iteratively generate ellipses then remove points within the ellipse
    
    Parameters:
    n_iter (int): Number of times to iterate
    '''
    for n in range(n_iter):
        axis_ratio, center, width, height, phi = plot_points(points, show_plots=show_plots)
        remove_arr = [remove_ellipse_point(point, width, height, phi) for point in points]
        remove_arr_idx = [idx for idx, true_val in enumerate(remove_arr) if not true_val] # Which indices we want to keep
        points = [points[index] for index in remove_arr_idx]
    return points



### Main funcitons

def generate_uniformheight_distribution(n_repeats = 1000):
    """Makes the distribution where all galaxies have a specified height"""
    # heights = [0.01, 0.25, 0.5, 0.75, 1.0]
    heights = [0.2]


    for height in heights:
        axis_ratios = []
        for i in range(n_repeats):
            points = make_shape(height=height)
            points = rotate_cylinder(points)
            points = iterate_remove_points(points, 3)
            axis_ratio, center, width, height_out, phi = plot_points(points)
            axis_ratios.append(axis_ratio)

            
        fig, ax = plt.subplots()
        ax.hist(axis_ratios, 20, color='black')

        ax.set_xlim(-0.05, 1.05) 

        fig.savefig(imd.axis_output_dir + f'/distribution_simulations/{height}.pdf')

def generate_mixedheight_distribution(n_repeats = 1000):
    """Makes the distribution where galaxies are a mix of heights"""

    # Input heights as a list of tuples of the form (height, fraction) where all the fractions add up to one
    height_tuples = [(0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2)]

    # Build a list of what number to change heights one
    cutoffs = [0]
    for i in range(len(height_tuples)):
        cutoffs.append(height_tuples[i][1] * 1000 + cutoffs[i])
    cutoffs = cutoffs[1:]

    axis_ratios = []
    count = 0 # Start counting the loops
    for i in range(n_repeats):
        idx = next(x for x, val in enumerate(cutoffs) if val > count) # gets element of list greater than count https://www.geeksforgeeks.org/python-get-the-index-of-first-element-greater-than-k/
        height = height_tuples[idx][0]
        points = make_shape(height=height)
        points = rotate_cylinder(points)
        points = iterate_remove_points(points, 3)
        axis_ratio, center, width, height_out, phi = plot_points(points)
        axis_ratios.append(axis_ratio)
        count = count + 1

        
    fig, ax = plt.subplots()
    ax.hist(axis_ratios, 20, color='black')

    ax.set_xlim(-0.05, 1.05)

    text_start = 0.95
    for i in range(len(height_tuples)):
        ax.text(0.05, text_start-0.04*i, f'Axis: {height_tuples[i][0]}, Pct: {height_tuples[i][1]}', transform=ax.transAxes, color='red')

    fig.savefig(imd.axis_output_dir + f'/distribution_simulations/Sample_sim_test.pdf')




def generate_oneplot(height=0.0001):
    '''Just makes one plot and shows it to the user'''
    
    points = make_shape(height=height)
    points = rotate_cylinder(points)
    points = iterate_remove_points(points, 3, show_plots=True)
    axis_ratio, center, width, height, phi = plot_points(points, show_plots=True)

        
    



### Tests

def test_ellipse():
    """ Test case that generates an ellipse with a known axis ratio and returns it
    """
    axis_ratio = 0.111
    b = 1  # Don't change b

    n_points = 1000

    def generate_point_ellipse(a, b):
        '''Generates a point on an ellipse'''
        # Generate points on the outside of a cylinder
        r = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2*np.pi)
        x = a * np.cos(theta) * r
        y = b * np.sin(theta) * r
        point = np.array([x, y, 0])
        return point
    
    points = []
    for i in range(n_points):
        points.append(generate_point_ellipse(axis_ratio, b))

    points = rotate_cylinder(points, only_rot_z = True)
    
    points = iterate_remove_points(points, 3)

    axis_ratio_computed, center, width, height, phi = plot_points(points, show_plots=True)

    print(f'Axis ratio: {axis_ratio}, computed Axis Ratio: {axis_ratio_computed}')

    
    


    


# generate_uniformheight_distribution()
# generate_mixedheight_distribution()
# generate_oneplot(height=0.4)
# test_ellipse()


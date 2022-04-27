# Sets up the ellipse shapes that we use for plotting
import matplotlib as mpl
from matplotlib.patches import Ellipse



def get_ellipse_shapes(x_axis_len, y_axis_len, shape, scale_factor=0.035):
    '''Will return the appropriate dimensions for the ellipse to plot

    x_axis_len (float): Length of the x-axis
    y_axis_len (float): Length of the y-axis   
    shape (str): Shape from summary df 
    scale_factor (float): Fractional width of the axis for point size
    '''

    ellipse_width = scale_factor*x_axis_len # Width of the ellipse, in axes coords
    ellipse_fracs = [0.3, 0.5, 1] 


    if shape == '+':
        ellipse_frac = ellipse_fracs[0]
    elif shape == 'd':
        ellipse_frac = ellipse_fracs[1]
    elif shape == 'o':
        ellipse_frac = ellipse_fracs[2]
    elif shape == 1.0:
        ellipse_frac = ellipse_fracs[2]

    y_height_factor = (ellipse_width/x_axis_len) * y_axis_len
    ellipse_height = y_height_factor*ellipse_frac
    
    return ellipse_width, ellipse_height



# ax.errorbar(x_cord, y_cord, yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker='None', color=rgba)
# ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))


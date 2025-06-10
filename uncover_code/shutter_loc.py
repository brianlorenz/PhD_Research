from uncover_read_data import read_slit_loc_cat
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import ascii
import math
import numpy as np
from find_slit_extraction import gaussian_func_with_cont


def plot_shutter_pos(ax, id_msa, wcs, paper=False):
    slit_loc_df = read_slit_loc_cat()
    slit_loc_rows = slit_loc_df[slit_loc_df['id_msa']==id_msa]

    extraction_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables_referee/extraction_df.csv').to_pandas()
    extraction_row = extraction_df[extraction_df['id_msa'] == id_msa]
    center = extraction_row['center'].iloc[0]
    offset = extraction_row['offset'].iloc[0]*(9/4)
    new_center = center+offset # Maybe a sign error on offset here
    fwhm = extraction_row['fwhm'].iloc[0]*(9/4)
    sig = extraction_row['sig'].iloc[0]*(9/4)
    amp = extraction_row['amp'].iloc[0]
    cont = extraction_row['cont_level'].iloc[0]
    min_pix = new_center - fwhm/2
    max_pix = new_center + fwhm/2
    include_0, frac_0_start, frac_0_end = check_region_0(min_pix, max_pix)
    include_1, frac_1_start, frac_1_end = check_region_1(min_pix, max_pix)
    include_2, frac_2_start, frac_2_end = check_region_2(min_pix, max_pix)
    pixels = np.arange(0, 27, 0.001)
    y_vals = gaussian_func_with_cont(pixels, 0, amp, sig, cont)
    y_vals = y_vals-cont
    y_vals = np.abs(y_vals/np.max(y_vals))
    if new_center < 9:
        region_new_center = 0
        _, new_center_frac, _ = check_region_0(new_center, new_center)
    elif new_center < 18:
        region_new_center = 1
        _, new_center_frac, _ = check_region_1(new_center, new_center)
    else:
        region_new_center = 2
        _, new_center_frac, _ = check_region_2(new_center, new_center)
    frac_center = new_center%9


    
        
    frac_dict = {
        'region_0_fracs': (frac_0_start, frac_0_end),
        'region_1_fracs': (frac_1_start, frac_1_end),
        'region_2_fracs': (frac_2_start, frac_2_end),
        'region_0_included': include_0,
        'region_1_included': include_1,
        'region_2_included': include_2,
        'center_region': region_new_center,
        'center_frac': new_center_frac
    }

    vertices_list = []
    for i in range(len(slit_loc_rows)):
        row = slit_loc_rows.iloc[i]
        shutter_strs = [row['SHUT_S_REGION_0'], row['SRC_SHUT_S_REGION_1'], row['SHUT_S_REGION_2']]
        region_num = 0
        new_vertices_all_lists = []
        for shutter_str in shutter_strs:
            box_coords = split_shutter_to_radec(shutter_str)
            pixel_locations = [wcs.world_to_pixel(coord) for coord in box_coords]

            # Define the corners of the rectangle
            x1, y1 = pixel_locations[0][0], pixel_locations[0][1]  # Bottom left corner
            x2, y2 = pixel_locations[1][0], pixel_locations[1][1]  # Bottom right corner
            x3, y3 = pixel_locations[2][0], pixel_locations[2][1]  # Top right corner
            x4, y4 = pixel_locations[3][0], pixel_locations[3][1]  # Top left corner


            vertices_list.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            dist_01 = get_distance(vertices_list[region_num][0], vertices_list[region_num][1])
            dist_02 = get_distance(vertices_list[region_num][0], vertices_list[region_num][2])
            dist_03 = get_distance(vertices_list[region_num][0], vertices_list[region_num][3])
            dist_list  = [dist_01, dist_02, dist_03]
            mid_value = np.median(dist_list)
            for index, element in enumerate(dist_list):
                if element == mid_value:
                    mid_vertex_index = index+1
            possible_vertices = [1,2,3]
            possible_vertices = [vertex for vertex in possible_vertices if vertex != mid_vertex_index]
            
            # Create a Polygon patch
            patch_color = 'cyan'
            if paper:
                patch_color = 'whitesmoke'
            rect = patches.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], closed=True, fill=False, color=patch_color)

            # Add the patch to the axes
            ax.add_patch(rect)
            
            if frac_dict[f'region_{region_num}_included']:
                frac_start = frac_dict[f'region_{region_num}_fracs'][0]
                frac_end = frac_dict[f'region_{region_num}_fracs'][1]
                new_vertex_1_start, new_vertex_1_end = get_fraction_up_line(vertices_list[region_num][0], vertices_list[region_num][mid_vertex_index], frac_start=frac_start, frac_end=frac_end)
                new_vertex_2_start, new_vertex_2_end = get_fraction_up_line(vertices_list[region_num][possible_vertices[0]], vertices_list[region_num][possible_vertices[1]], frac_start=frac_start, frac_end=frac_end)
                new_vertices_list = [new_vertex_1_start, new_vertex_2_start, new_vertex_2_end, new_vertex_1_end]
                rect2 = patches.Polygon(new_vertices_list, closed=True, fill=True, color='black', alpha=0.5)
                if paper == False:
                    ax.add_patch(rect2)
                if region_num == frac_dict[f'center_region']:
                    center_frac = frac_dict[f'center_frac']
                    center_vertex_1, _ = get_fraction_up_line(vertices_list[region_num][0], vertices_list[region_num][mid_vertex_index], frac_start=center_frac, frac_end=frac_end)
                    center_vertex_2, _ = get_fraction_up_line(vertices_list[region_num][possible_vertices[0]], vertices_list[region_num][possible_vertices[1]], frac_start=center_frac, frac_end=frac_end)
                    if paper == False:
                        ax.plot([center_vertex_1[0], center_vertex_2[0]], [center_vertex_1[1], center_vertex_2[1]], marker='o', ls='-', color='cyan')
            else:
                new_vertices_list = [(-99, -99), (-99, -99), (-99, -99), (-99, -99)]
            new_vertices_all_lists.append(new_vertices_list)
           

            
            region_num = region_num + 1
    
    
    gauss_info = [pixels, y_vals, center_vertex_1, center_vertex_2]
    return vertices_list, new_vertices_all_lists, gauss_info

def get_scale_factor(point, pixels, y_vals, center_vertex_1, center_vertex_2):
    """Computes distance from point to midline, then returns how much to scale the point for optimal extraction based on provided gaussian
    
    pixels (array): x values, at high resolution.
    y_vals (array): y values if you plug the x values into the gaussian centerd at 0, and scaled
    center_vertex1, center_vertex_2 (): Vertices along the ceenteral line
    """
    center_vertex_1_arr = np.array([float(center_vertex_1[0]), float(center_vertex_1[1])])
    center_vertex_2_arr = np.array([float(center_vertex_2[0]), float(center_vertex_2[1])])
    distance = get_distance_to_line(center_vertex_1_arr, center_vertex_2_arr, point)
    scale_factor = y_vals[np.argmin(np.abs(pixels-distance))]
    return scale_factor

def get_fraction_up_line(p1, p2, frac_start, frac_end):
    x1, y1 = p1
    x2, y2 = p2

    long_slope, long_intercept = line_from_points(p1, p2)

    min_x = np.min([x1, x2])
    max_x = np.max([x1, x2])

    x_diff = max_x - min_x
    x_new_start = np.array(min_x + x_diff*frac_start)
    x_new_end = np.array(min_x + x_diff*frac_end)

    y_new_start = np.array(long_slope*x_new_start + long_intercept)
    y_new_end = np.array(long_slope*x_new_end + long_intercept)
    new_vertex_start = (x_new_start, y_new_start)
    new_vertex_end = (x_new_end, y_new_end)
    return new_vertex_start, new_vertex_end


def get_distance_to_line(line_p1, line_p2, datapoint):
    distance = np.cross(line_p2-line_p1,datapoint-line_p1)/np.linalg.norm(line_p2-line_p1)
    return np.abs(distance)

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    if x2 - x1 == 0:
        return "Vertical line: x = {}".format(x1)
    
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    
    return m, c

def check_region_0(min_pix, max_pix):
    if min_pix > 9: 
        include_0 = False
        frac_0_start = -99
        frac_0_end = -99
    else: 
        include_0 = True
        if min_pix < 0:
            frac_0_start = 0
        else:
            frac_0_start = min_pix / 9
        if max_pix > 9:
            frac_0_end = 1
        else:
            frac_0_end = max_pix / 9
    return include_0, frac_0_start, frac_0_end

def check_region_1(min_pix, max_pix):
    if (min_pix < 18 and max_pix > 9): 
        include_1 = True
        if max_pix > 18:
            frac_1_end = 1
        else:
            frac_1_end = (max_pix%9) / 9
        if min_pix < 9:
            frac_1_start = 0
        else:
            frac_1_start = (min_pix%9) / 9
        
    else: 
        include_1 = False
        frac_1_start = -99
        frac_1_end = -99
        
    return include_1, frac_1_start, frac_1_end


def check_region_2(min_pix, max_pix):
    if max_pix < 18: 
        include_2 = False
        frac_2_start = -99
        frac_2_end = -99
    else: 
        include_2 = True
        if max_pix > 27:
            frac_2_end = 1
        else:
            frac_2_end = (max_pix%9) / 9
        if min_pix < 18:
            frac_2_start = 0
        else:
            frac_2_start = (min_pix%9) / 9
    return include_2, frac_2_start, frac_2_end

def get_distance(point1, point2):
    distance = math.hypot(point2[0] - point1[0], point2[1] - point1[1])
    return distance

def check_point_in_shutter(x, y, vertices):
    in_shape = test_point(x, y, vertices)
    return in_shape

def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c >= 0

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right


def split_shutter_to_radec(shutter_str):
    split_str = shutter_str.split()
    box_coords = []
    for i in range(4):
        coord = SkyCoord(ra=float(split_str[2*i+2])*u.degree, dec=float(split_str[2*i+3])*u.degree)
        box_coords.append(coord)  
    return box_coords

if __name__ == "__main__":
    # plot_shutter_pos(1, 47875)
    pass
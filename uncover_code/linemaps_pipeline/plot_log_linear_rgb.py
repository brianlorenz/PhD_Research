"""
Combine 3 images to produce a properly-scaled RGB images 
with either log or linear scaling. 

Complementary to (and code structure inspired by) 
astropy.visualization.make_lupton_rgb. 

The three images must be aligned and have the same pixel scale and size.
"""

import numpy as np

from astropy.visualization import LinearStretch, LogStretch, \
                                  ManualInterval, ImageNormalize

_OUTPUT_IMAGE_FORMATS = [np.float64, np.uint8]

__all__ = ["make_log_rgb", "make_linear_rgb"]

class ImageMapping:
    """
    Class to map red, blue, green images into either
    a normalized float or an 8-bit image,
    by performing optional clipping and applying a scaling function.

    Parameters
    ----------
    minimum : float or array-like, shape(3), optional
        Intensity that should be mapped to black
        (a scalar or array for R, G, B).
    maximum : float or array-like, shape(3), optional
        Intensity that should be mapped to white
        (a scalar or array for R, G, B).
    """

    def __init__(self, minimum=None, maximum=None, stretch=LinearStretch):
        self._uint8max = float(np.iinfo(np.uint8).max)

        try:
            len(minimum)
        except TypeError:
            minimum = 3 * [minimum]
        if len(minimum) != 3:
            raise ValueError("please provide 1 or 3 values for minimum.")

        try:
            len(maximum)
        except TypeError:
            maximum = 3 * [maximum]
        if len(maximum) != 3:
            raise ValueError("please provide 1 or 3 values for maximum.")

        intervals = []
        for i in range(3):
            intervals.append(ManualInterval(vmin=minimum[i], vmax=maximum[i]))

        self.intervals = intervals
        self.stretch = stretch

    def make_rgb_image(self, image_r, image_g, image_b, output_image_format=np.float64):
        """
        Convert 3 arrays, image_r, image_g, and image_b into a RGB image,
        either as an 8-bit per-channel or normalized image.

        The input images can be int or float, and in any range or bit-depth,
        but must have the same shape (NxM).

        Parameters
        ----------
        image_r : ndarray
            Image to map to red.
        image_g : ndarray
            Image to map to green.
        image_b : ndarray
            Image to map to blue.
        output_image_format : numpy scalar type, optional
            Image output format

        Returns
        -------
        RGBimage : ndarray
            RGB color image with the specified format as an NxMx3 numpy array.
        """
        if output_image_format not in _OUTPUT_IMAGE_FORMATS:
            raise ValueError(
                f"'output_image_format' must be one of {_OUTPUT_IMAGE_FORMATS}!"
            )

        image_r = np.asarray(image_r)
        image_g = np.asarray(image_g)
        image_b = np.asarray(image_b)

        if (image_r.shape != image_g.shape) or (image_g.shape != image_b.shape):
            msg = "The image shapes must match. r: {}, g: {} b: {}"
            raise ValueError(msg.format(image_r.shape, image_g.shape, image_b.shape))

        if np.issubdtype(output_image_format, float):
            conv_images = self._convert_images_to_float(image_r, image_g, image_b)
        elif np.issubdtype(output_image_format, np.uint8):
            conv_images = self._convert_images_to_uint8(image_r, image_g, image_b)

        return np.dstack(conv_images).astype(output_image_format)

    def apply_mappings(self, image_r, image_g, image_b):
        """Apply mapping stretch and intervals"""
        image_rgb = [image_r, image_g, image_b]
        for i, img in enumerate(image_rgb):
            norm = ImageNormalize(img, interval=self.intervals[i],
                                  stretch=self.stretch, clip=True)
            img = norm(img)
            image_rgb[i] = img
        return np.array(image_rgb)

    def _convert_images_to_float(self, image_r, image_g, image_b):
        """
        Use the mapping to convert images image_r, image_g, and image_b
        to a triplet of normalized float images.
        """
        image_rgb = self.apply_mappings(image_r, image_g, image_b)
        return image_rgb.astype(np.float64)

    def _convert_images_to_uint8(self, image_r, image_g, image_b):
        """
        Use the mapping to convert images image_r, image_g, and image_b
        to a triplet of uint8 images.
        """
        image_rgb = self.apply_mappings(image_r, image_g, image_b)
        pixmax = self._uint8max
        image_rgb *= pixmax
        return image_rgb.astype(np.uint8)



def make_log_rgb(
    image_r,
    image_g,
    image_b,
    minimum=None,
    maximum=None,
    scalea=1000,
    filename=None,
    output_image_format=np.float64,
):
    """
    Return a Red/Green/Blue color image from 3 images using a log stretch,
    with optional clipping of the input values before scaling, using

    .. math::

        y = \frac{\log{(a x + 1)}}{\log{(a + 1)}}

    The input images can be int or float, and in any range or bit-depth,
    but must have the same shape (NxM).

    For a more detailed look at the use of this method, see the document
    :ref:`astropy:astropy-visualization-rgb`.

    Parameters
    ----------
    image_r : ndarray
        Image to map to red.
    image_g : ndarray
        Image to map to green.
    image_b : ndarray
        Image to map to blue.
    minimum : float or array-like, shape(3), optional
        Intensity that should be mapped to black (a scalar or array for R, G, B).
        If `None`, each image's minimum value is used.
    maximum : float or array-like, shape(3), optional
        Intensity that should be mapped to white (a scalar or array for R, G, B).
        If `None`, each image's maximum value is used.
    scalea : float, optional
        Log scaling exponent.
    filename : str, optional
        Write the resulting RGB image to a file (file type determined
        from extension).
    output_image_format : numpy scalar type, optional
        Image output format

    Returns
    -------
    rgb : ndarray
        RGB (either float or integer with 8-bits per channel) color image
        as an NxMx3 numpy array.

    Notes
    -----
    This procedure of clipping and then scaling is similar to the DS9
    image algorithm (see the DS9 reference guide [1]_).

    References
    ----------
    .. [1] http://ds9.si.edu/doc/ref/how.html
    """
    log_map = ImageMapping(minimum=minimum, maximum=maximum,
                           stretch=LogStretch(a=scalea))
    rgb = log_map.make_rgb_image(
        image_r, image_g, image_b, output_image_format=output_image_format
    )

    if filename:
        import matplotlib.image

        matplotlib.image.imsave(filename, rgb, origin="lower")

    return rgb



def make_linear_rgb(
    image_r,
    image_g,
    image_b,
    minimum=None,
    maximum=None,
    filename=None,
    output_image_format=np.float64,
):
    """
    Return a Red/Green/Blue color image from 3 images using a linear stretch,
    with optional clipping of the input values before scaling.

    The input images can be int or float, and in any range or bit-depth,
    but must have the same shape (NxM).

    For a more detailed look at the use of this method, see the document
    :ref:`astropy:astropy-visualization-rgb`.

    Parameters
    ----------
    image_r : ndarray
        Image to map to red.
    image_g : ndarray
        Image to map to green.
    image_b : ndarray
        Image to map to blue.
    minimum : float or array-like, shape(3), optional
        Intensity that should be mapped to black (a scalar or array for R, G, B).
        If `None`, each image's minimum value is used.
    maximum : float or array-like, shape(3), optional
        Intensity that should be mapped to white (a scalar or array for R, G, B).
        If `None`, each image's maximum value is used.
    filename : str, optional
        Write the resulting RGB image to a file (file type determined
        from extension).
    output_image_format : numpy scalar type, optional
        Image output format

    Returns
    -------
    rgb : ndarray
        RGB (either float or integer with 8-bits per channel) color image
        as an NxMx3 numpy array.

    Notes
    -----
    This procedure of clipping and then scaling is similar to the DS9
    image algorithm (see the DS9 reference guide [1]_)

    References
    ----------
    .. [1] http://ds9.si.edu/doc/ref/how.html
    """
    linear_map = ImageMapping(minimum=minimum, maximum=maximum,
                              stretch=LinearStretch())
    rgb = linear_map.make_rgb_image(
        image_r, image_g, image_b, output_image_format=output_image_format
    )

    if filename:
        import matplotlib.image

        matplotlib.image.imsave(filename, rgb, origin="lower")

    return rgb

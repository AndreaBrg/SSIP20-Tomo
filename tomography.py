from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage._shared.fft import fftmodule
from skimage._shared.utils import convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.io import imread
from skimage.transform import warp
from skimage.transform.radon_transform import _get_fourier_filter

if fftmodule is np.fft:
    # fallback from scipy.fft to scipy.fftpack instead of numpy.fft
    # (fftpack preserves single precision while numpy.fft does not)
    from scipy.fftpack import fft, ifft
else:
    fft = fftmodule.fft
    ifft = fftmodule.ifft


def generate_synthetic_data(size):
    """ Synthetic binary data """
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:size, 0:size]
    mask_outer = (x - size / 2.) ** 2 + (y - size / 2.) ** 2 < (size / 2.) ** 2
    mask = np.zeros((size, size))
    points = size * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=size / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))


def build_sinogram(image, angles=None):
    """
    Returns an image containing radon transform (sinogram).
    Each column of the image corresponds to a projection along a different angle.
    The tomography rotation axis lies at the pixel index ``sinogram.shape[0] // 2``
    along the 0th dimension of ``sinogram``.
    """
    image = convert_to_float(image, True)
    
    # Center of the image, rotation pivot
    center = image.shape[0] // 2
    
    # Output sinogram, each column contains, for a given angle, the sum of input image's pixels row-by-row
    sinogram = np.zeros((image.shape[0], len(angles)))
    
    # Build the sinogram
    for i, angle in enumerate(np.deg2rad(angles)):
        # Get angle cos and sin
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        R = np.array([[cos_a,   sin_a,  -center * (cos_a + sin_a - 1)],
                      [-sin_a,  cos_a,  -center * (cos_a - sin_a - 1)],
                      [0,       0,      1]])
        # Apply rotation to the image
        rotated = warp(image, R, clip=False)
        # Define the sinogram column for the current angle i.e. the sum of the rotated image's rows
        sinogram[:, i] = rotated.sum(0)
    return sinogram


def filtered_back_projection(radon_image, theta=None, filter=None, interpolation="linear"):
    img_shape = radon_image.shape[0]
    output_size = image.shape[0]
    
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    
    # Apply filter in Fourier domain
    if filter is not None:
        fourier_filter = _get_fourier_filter(projection_size_padded, filter)
        projection = fft(img, axis=0) * fourier_filter
        radon_image = np.real(ifft(projection, axis=0)[:img_shape, :])
    
    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    
    for col, angle in zip(radon_image.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        else:
            interpolant = interp1d(x, col, kind=interpolation,
                                   bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)
    
    # Normalization in [0, 1]
    return (reconstructed - np.min(reconstructed)) / (np.max(reconstructed) - np.min(reconstructed))
    # return reconstructed * np.pi / (2 * len(theta))


if __name__ == '__main__':
    """
    GET IMAGE
    Either generate a random one or read from folder
    """
    
    # Generate square image of size X size pixels
    # size = 512
    # image = generate_synthetic_data(size)
    
    # Read image from images folder
    # image = imread("images/image2.png", as_gray=True)
    
    # Use standard test image
    image = shepp_logan_phantom()
    
    plt.imshow(image, cmap=plt.cm.Greys_r)
    plt.show()
    
    """
    CREATE THE SINOGRAM
    Define the amount of angles to use for the rotation in (0, 180)
    """
    
    angles_amount = max(image.shape)  # change to desired amount of angles
    angles = np.linspace(0., 180., angles_amount, endpoint=False)
    
    sinogram = build_sinogram(image, angles)
    
    plt.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    plt.show()
    
    """
    RECONSTRUCT THE ORIGINAL IMAGE
    WITH FILTER BACK PROJECTION
    Define filter and interpolation procedures as needed
    """
    
    # filter_types = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
    # interpolation_types = ['linear', 'nearest', 'cubic']
    reconstruction = filtered_back_projection(sinogram, angles, filter="cosine")
    plt.imshow(reconstruction, cmap=plt.cm.Greys_r)
    plt.show()

    # Evaluate the reconstruction error
    error = reconstruction - image
    print("FBP rms reconstruction error: {:.3g}".format(np.sqrt(np.mean(error ** 2))))
    plt.imshow(error, cmap=plt.cm.Greys_r)
    plt.show()

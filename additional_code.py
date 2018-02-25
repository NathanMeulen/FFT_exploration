import numpy as np
import matplotlib.pyplot as plt

def ellipse(x, y, center, major_axis, minor_axis, theta, fill_value):
    '''
    Returns the value of an ellipse in space for the specified coordinates
    
    Reference: 
        -https://math.stackexchange.com/questions/426150/
          what-is-the-general-equation-of-the-ellipse-that-
          is-not-in-the-origin-and-rotate
    
    Parameters:
        x - (array-like) x coordinate(s) at which it is to be evaluated
        y - (array-like) y coordinate(s) at which it is to be evaluated
        center - (tuple) the center of the ellipse to be drawn
        major_axis - (float) major (x) axis of the ellipse
        minor_axis - (float) minor (y) axis of the ellipse
        theta - (float) In rad. Angle of rotation relative to the y axis
        fill_value - (float) value to return if points are inside ellipse
    Returns:
        grid - (array-like) grid of zeros and fill_value, depending on which
                    points fall within the ellipse
    '''
    grid = np.zeros(np.array(x).shape)
    mask = np.power(((x-center[0])*np.cos(np.radians(theta))+(y-center[1]) \
             *np.sin(np.radians(theta)))/major_axis, 2) + \
             np.power(((x-center[0])*np.sin(np.radians(theta))-(y-center[1]) \
             *np.cos(np.radians(theta)))/minor_axis , 2) <= 1
    grid[mask] = fill_value
    return grid

def shepp_logan_phantom(x, y):
    '''
    Returns an image of the Shepp-Logan phantom centered around the origin
    
    Reference: 
        -https://en.wikipedia.org/wiki/Shepp–Logan_phantom
    
    Parameters:
        x - (array-like) x coordinate(s) at which it is to be evaluated
        y - (array-like) y coordinate(s) at which it is to be evaluated
    Returns:
        grid - (array-like) grid of zeros and superposition of ellipse values, 
                    depending on which points fall within the ellipse
    '''
    
    # Define the ellipses that make up the Shepp-Logan phantom
    #                    center, maj. a., min. a., theta,fill_value
    ellipses = [[         (0,0),    0.69,    0.92,     0,       2],
                [   (0,-0.0184),  0.6624,   0.874,     0,   -0.98],
                [      (0.22,0),    0.11,    0.31,   -18,   -0.02],
                [     (-0.22,0),    0.16,    0.41,    18,   -0.02],
                [      (0,0.35),    0.21,    0.25,     0,    0.01],
                [       (0,0.1),   0.046,   0.046,     0,    0.01],
                [      (0,-0.1),   0.046,   0.046,     0,    0.01],
                [(-0.08,-0.605),   0.046,   0.023,     0,    0.01],
                [    (0,-0.605),   0.023,   0.023,     0,    0.01],
                [ (0.06,-0.605),   0.023,   0.046,     0,    0.01]]
    grid = np.zeros(np.array(x).shape)
    
    for e in ellipses:
        grid += ellipse(x, y, *e)
        
    return np.array(grid)

def tomographic_projection(offset_xprime, phi, fn, nx=599, ny=601, full_set=False, verbose=False):
    '''
    Takes the line integral along a specific function rotated at a specified
    angle.
    
    Reference: 
        -M. Pettyjohn
    
    Parameters:
        offset_xprime - (array-like) offset(s) from the origin at which to take
                            the line integral
        phi - (float) In degrees. Angle at which to rotate the function
        fn - (function) function to take the tomographic projection of
        nx - (int) Optional. x resolution of the grid over which to evaluate
        ny - (int) Optional. y resolution of the grid over which to evaluate
        full_set - (boolean) Optional. Set to true to force evaluation over all
                        possible angles and avoid truncation errors
        verbose - (boolean) Optional. Set to true to also have the image of
                        the projection returned
    Returns:
        line_int - (array-like) line integral ove all specified offsets
    '''
    x, y = np.linspace(-1.0,+1.0,nx), np.linspace(-1.0,+1.0,ny)
    xx, yy = np.meshgrid(x,y)
    phi = np.radians(phi)
    
    # Rotate image
    x_prime = (xx)*np.cos(phi) + yy*np.sin(phi)
    y_prime = -(xx)*np.sin(phi) + yy*np.cos(phi)
    
    grid = shepp_logan_phantom(x_prime, y_prime)
    
    #Calculate indexes for all offsets
    line_index = np.intc(xx.shape[0]//2 + offset_xprime*nx//2 - 1) \
                if not full_set else np.arange(x.size)
    #Calculate the discrete integral for an arbitrary number of lines
    line_int = np.sum(grid[:, line_index], axis=0) \
                if np.array(line_index).size != 1 \
                else np.sum(grid[:, line_index])
    
    if verbose:
        grid[:, line_index] = 3
        return line_int, grid
    else:
        return line_int
    
def sinogram(offset_xprime, angle_phi, nx=201, ny=281, full_set=False):
    '''
    Also known as the Radon Transform. Generates a sinogram of the sheep-logan
    phantom.
    
    Parameters:
        offset_xprime - (array-like) offset(s) from the origin at which to take
                            the line integral
        angle_phi - (array-like) In rad. Angles at which to rotate the function
        nx - (int) Optional. x resolution of the grid over which to evaluate
        ny - (int) Optional. y resolution of the grid over which to evaluate
        full_set - (boolean) Optional. Set to true to force evaluation over all
                        possible angles and avoid truncation errors
    Returns:
        grid - (array-like) sinogram image
    '''
    offset_xprime, angle_phi = np.array(offset_xprime), np.array(angle_phi)
    grid = np.zeros((offset_xprime.size, angle_phi.size)) if not full_set else\
                np.zeros((nx, angle_phi.size))
    
    for i, a in enumerate(angle_phi):
        grid[:,i] = tomographic_projection( offset_xprime, np.degrees(a), 
                        shepp_logan_phantom, full_set=full_set, nx=nx, ny=ny)
    return grid

def backprojection(sinogram, angle_phi, output_size=None, filtered=True):
    '''
    Reconstructs an image from a sinogram for a set number of angles.
    
    References:
    - P. Ciunkiewicz
    - M. Pettyjohn
    - github.com/scikit-image/scikit-image/blob/master/
        skimage/transform/radon_transform.py
    
    Parameters:
        sinogram - (array-like) sinogram image
        angle_phi - (array-like) In rad. Angles at which to project the image
        output_size - (int) Optional. Forces resolution of returned image if 
                        smaller than initial estimate
    Returns:
        reconstructed - (array-like) reconstructed image
    '''
    num_offsets = sinogram.shape[0]
    output_size = sinogram.shape[0] if output_size == None else \
                                    min(output_size, sinogram.shape[0])
    middle = output_size // 2
    reconstructed = np.zeros((output_size, output_size))
    xx, yy = np.mgrid[-middle:(middle + 1), -middle:(middle + 1)]
    
    if filtered == True:
        # Ramp filter
        fourier_filter = 2 * np.abs(np.fft.fftfreq(num_offsets).reshape(-1, 1))
        sinogram_fft = np.fft.fft(sinogram, axis=0) * fourier_filter
        filtered_sinogram = np.fft.ifft(sinogram_fft, axis=0).real
    else: filtered_sinogram = sinogram

    # Interpolate the whole grid for each angle from scan line
    for i in range(len(angle_phi)):
        # Rotated coordinates for correct superposition
        kx_prime = yy*np.cos(angle_phi[i]) - xx*np.sin(angle_phi[i])
        backprojected = np.interp(kx_prime, xx[:output_size,0], filtered_sinogram[:output_size, i], left=0, right=0)

        reconstructed += backprojected
    
    reconstructed[reconstructed < 0] = 0 # Filters out erronous approximations
    return reconstructed * 2 / len(angle_phi)

def residual(matrix1, matrix2):
    '''
    Finds the residual of two matrices of the same size.
    
    Parameters:
        matrix1 - (array-like) first matrix
        matrix2 - (array-like) second matrix
    Returns:
        - (float) residual
    '''
    difference = np.abs(matrix1-matrix2)
    return np.sum(difference)

def make_plot(title='', x_label='', y_label='', z_label='', x_lim=(0, 0), y_lim=(0,0), z_lim=(0,0), grid=True, size=(8, 5), dpi=115, three_d=False):
    '''
    Generates a plot object
    
    Parameters:
        title - (string) title of the plot
        x - (float) a list of values for omega/omega_0
        y - (float) a list of values for the amplitude
        xlim - (float) the limit for the x axis, this is a list containing 2 variables, 
                    the max and min value of x
        ylim - (float) the limit for the y axis, this is a list containing 2 variables, 
                    the max and min value of y
        size - (int) resizing the figure of the plot
        dpi - (int) increasing the quality of the image produced
    Returns:
        plot - (object) the plot
    '''
    fig = plt.figure(figsize=(size[0], size[1]), dpi=dpi, facecolor='0.9')
    plot = fig.add_subplot(111) if not three_d else fig.add_subplot(111, projection='3d')
    
    plot.set_title(title)
    if y_lim[0] != 0 and y_lim[1] !=0: plot.set_ylim(y_lim[0], y_lim[1])
    if x_lim[0] != 0 and x_lim[1] !=0: plot.set_xlim(x_lim[0], x_lim[1])
    if z_lim[0] != 0 and z_lim[1] !=0: plot.set_zlim(z_lim[0], z_lim[1])
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    if three_d: plot.set_zlabel(z_label)
    if grid: plt.grid()
    
    return plot

def table_gen(title='', headers=[], data=[], aspect=1.0, rounding=4):
    '''
    Generates a matplotlib table
    
    Credit: P. Ciunkiewicz
    
    Parameters:
        title - (string) title of the table
        headers - (list or tuple) table column headers
        data - (array) array of data values for the table, first dimension must match headers
        aspect - (float) vertical scaling of the table, y:x aspect ratio
        rounding - (int) number of digits to round data values to before display
    Returns:
        none
    '''
    
    fig = plt.figure(figsize=(8,4), facecolor='0.9')
    ax = plt.subplot()
    
    content = np.round(data, rounding)
    labels = headers
    table = ax.table(cellText=(content), colLabels=(labels), loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, aspect)
    ax.set_title(title)
    ax.axis('off')
    
    fig.tight_layout()
    plt.show()
    
    return

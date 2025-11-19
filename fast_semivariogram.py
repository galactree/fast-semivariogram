'''
by Tree Smith

Provides an implementation of a fast semivariogram algorithm for a zero mean spatially varying field.

'''

import numpy as np
import scipy


def fast_semivariogram(data_grid, box_size, bins=50, separation_cutoff=None, return_bin_centres = True):
    '''
    Implementation of a fast semivariogram for a zero mean spatially varying field. The function
    uses the fast Fourier transform to calculate the variance between all pairs of points within the
    field.


    Parameters
        data_grid : ndarray, shape (N, M)
            two dimensional regularly spaced grid of data values.

        box_size: array_like, shape (2,)
            physical size of grid in x and y directions. 
            The first element corresponds to the physical size of the first dimension and the second
            element corresponds to the physical size of the second dimension.
            i.e. box_size[0] * N is the resolution of or the physical space between the elements in data_grid
        
        bins: int or sequence, optional
            Separation bins to use for the semivariogram computation. If bins is an int, it defines the number of 
            equal-width separation bins. If bins is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths. The default is 50.
        

        separation_cutoff: float, optional
            Upper limit of the separations to compute the semivariogram. If None provided, the maximum physical separation available
            is used which may lead to undesirable results. Default is None.
        
        return_bin_centres : bool, optional
            If True, the centre of each separation bins is returned. If bins is a sequence with non-uniform bin widths and 
            return_bin_centres is True, ValueError will be raised. If False, the bin edges are returned.
  
    Returns
        bin_centres : nd.array, optional
            Returned only if return_bin_centres is True and if bins is an int or a sequence with uniform bin widths. 
            The centres of the separation bins which the semivariogram is computed over, shape (len(semivariogram_values),)
        
        bin_edges : nd.array, optional
            Returned only if return_bin_centres is False. The edges of the separation bins which the semivariogram is 
            computed over. shape, len(semivariogram_values) + 1.
            If the bins parameter is a sequence, the returned bin_edges will be the exact same as the bins parameter.
        
        semivariogram_values : nd.array
            The values of the semivariogram function in each of the separation bins.
        
        counts : nd.array
            The number of pairs of points in each separation bin.

    '''


    # set up steps (mask and padding)
    nx, ny = data_grid.shape # shape
    pad_shape =(2*nx -1, 2*ny-1) #required padding
    M = (np.isnan(data_grid)) # mask
    data_copy = np.zeros_like(data_grid) # create a copy of the original array so it is unchanged by semivariogram computation

    data_copy[~M] = data_grid[~M] # set nans to 0
    M = (~M).astype(float) # 1 if non-nan

    lag_y = np.arange(pad_shape[0]) - (data_grid.shape[0] - 1)  # vertical shift (rows)
    lag_x = np.arange(pad_shape[1]) - (data_grid.shape[1] - 1)  # horizontal shift (cols)


    lag_Y, lag_X = np.meshgrid(lag_x, lag_y) # possible xy lag pairs

    # convert to physical units
    lag_X = (box_size[0]/nx) * lag_X

    lag_Y = (box_size[1]/ny) * lag_Y

    r = (lag_X**2 + lag_Y**2)**0.5 # total lag distance


    # calculate the variogram values using fft convolutions:
    gamma = scipy.signal.fftconvolve(M, (M*data_copy**2)[::-1, ::-1], mode='full') + scipy.signal.fftconvolve((M*data_copy**2), M[::-1, ::-1], mode='full') - 2*scipy.signal.fftconvolve((M*data_copy), (M*data_copy)[::-1, ::-1])


    N = scipy.signal.fftconvolve(M, M[::-1, ::-1], mode='full') # normalisation

    # Calculate bin_edges from bins, separation_cutoff and bin_size
    # if bins is list_like, use the input as the bin_edges for the separations
    bins = np.array([bins]).flatten()
    if len(bins) > 1 :
        bin_edges = bins
                

    else:
        bins = bins[0]

        if separation_cutoff is None:
            separation_cutoff = r.max()

        bin_edges = np.linspace(0, separation_cutoff, bins+1)

                
        
    # bin by total lag distance
    semivariogram_info = scipy.stats.binned_statistic(r.flatten(), gamma.flatten(), statistic=np.nansum, bins=bin_edges).statistic


    counts =  scipy.stats.binned_statistic(r.flatten(), N.flatten(), statistic=np.nansum, bins=bin_edges).statistic # bin normalisation

    semivariogram_values = 0.5*(semivariogram_info/counts) # compute semivariogram

    if return_bin_centres:

        # calculate centre of each bin from the bin_edges
        bin_centres = (bin_edges[1:] + bin_edges[:-1])/2

        return bin_centres, semivariogram_values, counts
    
    else:
        return bin_edges, semivariogram_values, counts



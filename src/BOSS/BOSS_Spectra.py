import numpy as np
from astropy.io import fits
from pathlib import Path
from definitions import PROJECT_ROOT


def read_fits(filename):
    '''
    Columns:
        0: Wavelength (Ang)
        1: Flux (calibrated)
        2: Variance
        3: Spectral resolution
    '''
    with fits.open(filename) as file:
        data = np.array([[d[1], d[0], 1/d[2], d[5]] for d in file[1].data])
    data[:,0] = 10**data[:,0]
    data = data[(data[:,0] >= 3750.00) & (data[:,0] <= 8995.63)]
    lamost = np.loadtxt(Path(__file__).parent / 'LAMOST_resolution.txt', skiprows=5)
    data[:,3] = np.interp(data[:,0], lamost[:,0], lamost[:,1])
    '''# Cleaning up dispersion channel
    i = 0
    i_last_ok = -1
    while i < len(data):
        if data[i, 3] > 0.1:
            if i_last_ok < i-1 and i_last_ok >= 0:
                # We have found the next valid pixel, interpolate the invalid area to clean up
                # In case of invalid pixels at the start of the spectrum, we need to handle
                # those separately
                data[i_last_ok+1:i, 3] = np.interp(
                    data[i_last_ok+1:i, 0],
                    np.array([data[i_last_ok,0], data[i,0]]),
                    np.array([data[i_last_ok,3], data[i,3]])
                )
            i_last_ok = i
        else:
            data[:,2] = np.infty
        i += 1
    # Cleaning up dispersion channel: edge cases
    if data[0,3] < 0.1:
        i_first_ok = 0
        while data[i_first_ok,3] < 0.1 and i_first_ok < len(data):
            i += 1
        data[:i_first_ok,3] = np.interp(
            data[:i_first_ok, 0],
            np.array([data[i_first_ok,0], data[i_first_ok+1,0]]),
            np.array([data[i_first_ok,3], data[i_first_ok+1,3]])
        )
    if data[-1,3] < 0.1:
        # We can reuse the i_last_ok from before since it hasn't changed since
        data[i_last_ok:,3] = np.interp(
            data[i_last_ok:, 0],
            np.array([data[i_last_ok-1,0], data[i_last_ok,0]]),
            np.array([data[i_last_ok-1,3], data[i_last_ok,3]])
        )'''
        
    return data



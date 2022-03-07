# -*- coding: utf-8 -*-
#
# Spectral estimation with (multi-)tapered FFT
#

# Builtin/3rd party package imports
import numpy as np
from scipy import signal


def mtmfft(data_arr,
           samplerate,
           nSamples=None,
           taper="hann",
           taper_opt=None,
           demean_taper=False):
    """
    (Multi-)tapered fast Fourier transform. Returns
    full complex Fourier transform for each taper.
    Multi-tapering only supported with Slepian windwows (`taper="dpss"`).

    Parameters
    ----------
    data_arr : (N,) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate in Hz
    nSamples : int or None
        Absolute length of the (potentially to be padded) signals
        or `None` for no padding (`N` is the number of samples)
    taper : str or None
        Taper function to use, one of `scipy.signal.windows`
        Set to `None` for no tapering.
    taper_opt : dict or None
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with ``taper='dpss'`` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    demean_taper : bool
        Set to `True` to perform de-meaning after tapering

    Returns
    -------
    ftr : 3D :class:`numpy.ndarray`
         Complex output has shape ``(nTapers x nFreq x nChannels)``.
    freqs : 1D :class:`numpy.ndarray`
         Array of Fourier frequencies

    Notes
    -----
    For a (MTM) power spectral estimate average the absolute squared
    transforms across tapers:

    ``Sxx = np.real(ftr * ftr.conj()).mean(axis=0)``

    The FFT result is normalized such that this yields the squared amplitudes.
    """

    # attach dummy channel axis in case only a
    # single signal/channel is the input
    if data_arr.ndim < 2:
        data_arr = data_arr[:, np.newaxis]

    signal_length = data_arr.shape[0]
    if nSamples is None:
        nSamples = signal_length

    nChannels = data_arr.shape[1]

    freqs = np.fft.rfftfreq(nSamples, 1 / samplerate)
    nFreq = freqs.size
    # frequency bins
    dFreq = freqs[1] - freqs[0]

    # no taper is boxcar
    if taper is None:
        taper = 'boxcar'

    if taper_opt is None:
        taper_opt = {}

    taper_func = getattr(signal.windows, taper)
    # only really 2d if taper='dpss' with Kmax > 1
    # here we take the actual signal lengths!
    windows = np.atleast_2d(taper_func(signal_length, **taper_opt))

    # only(!!) slepian windows are already normalized
    # per pedes L2 normalisation for all other tapers

    if taper == 'dpss':
        windows = np.sqrt(2 / dFreq) * windows
    # weird 3 point normalization,
    # scipy's hann is NOT [.25, .5, .25] in Fourier space
    elif taper in ('hann', 'hamming'):
        windows = np.sqrt(8 / 3) * windows / np.sqrt(windows.sum() * dFreq)
    # boxcar has full length integral
    elif taper == 'boxcar':
        windows = windows / np.sqrt(signal_length / 2 * dFreq)
    else:
        windows = np.sqrt(2) * windows / np.sqrt(windows.sum() * dFreq)

    # Fourier transforms (nTapers x nFreq x nChannels)
    ftr = np.zeros((windows.shape[0], nFreq, nChannels), dtype='complex64')

    for taperIdx, win in enumerate(windows):
        win = np.tile(win, (nChannels, 1)).T
        win *= data_arr
        # de-mean again after tapering - needed for Granger!
        if demean_taper:
            win -= win.mean(axis=0)
        # real fft takes only 'half the energy'/positive frequencies,
        # multiply by sqrt of 2 to correct for this
        ftr[taperIdx] = np.sqrt(2) * np.fft.rfft(win, n=nSamples, axis=0)
        # normalization
        ftr[taperIdx] /= np.sqrt(nSamples)

    return ftr, freqs

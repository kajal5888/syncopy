# -*- coding: utf-8 -*-
#
# Pairwise Phase Consistency measure (Vinck 2010)
#

# Builtin/3rd party package imports
import numpy as np

# syncopy imports
from syncopy.specest.mtmfft import mtmfft
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import spectralConversions


def rel_phases(trl_dat,
               samplerate=1,
               nSamples=None,
               taper="hann",
               taper_opt=None,
               fullOutput=False):

    """
    Calculate relative phases between all
    (``nChannels x nChannels + 1) / 2``) unique channel
    combinations as needed for the
    pairwise phase consistency as outlined in Ref. [1]_

    Parameters
    ----------
    trl_dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    samplerate : float
        Samplerate in Hz
    nSamples : int or None
        Absolute length of the (potentially to be padded) signals
        or `None` for no padding (`N` is the number of samples)
    taper : str or None
        Taper function to use, one of :module:`scipy.signal.windows`
        Set to `None` for no tapering.
    taper_opt : dict, optional
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with ``taper = 'dpss'`` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    fullOutput : bool
        For backend testing or stand-alone applications, set to `True`
        to return also the `freqs` array.

    Returns
    -------
    CS_ij : (nFreq, K, K) :class:`numpy.ndarray`
        Relative phases for all channel combinations ``i,j``,
        `K` corresponds to number of input channels.

    freqs : (nFreq,) :class:`numpy.ndarray`
        The Fourier frequencies if ``fullOutput = True``

    See also
    --------
    mtmfft : :func:`~syncopy.specest.mtmfft.mtmfft`
             (Multi-)tapered Fourier analysis

    Notes
    -----
    .. [1] Vinck, Martin, et al. "The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization." Neuroimage 51.1 (2010): 112-122
    """

    # compute the individual spectra
    # specs have shape (nTapers x nFreq x nChannels)
    specs, freqs = mtmfft(trl_dat, samplerate, nSamples, taper, taper_opt)

    # extract phases
    phases = np.angle(specs)

    # arithmetic phase differences
    pdiff = phases[:, :, np.newaxis, :] - phases[:, :, :, np.newaxis]
    # actual phase differences on the unit circle
    # \Delta phi = arctan y/x
    pdiff = np.arctan2(np.sin(pdiff), np.cos(pdiff))

    # now average tapers, these are the 'relative phases'
    # between all channels
    pdiff = np.mean(pdiff, axis=0)

    return pdiff, freqs


nSamples = 1000
nTrials = 10
fs = 1000
f1 = 40
sig1 = np.cos(np.arange(nSamples) * 1 / fs * 2 * np.pi * f1)

rel_ph = np.zeros((nSamples // 2 + 1, 2, 2, nTrials))

for i in range(nTrials):
    noise = np.random.randn(nSamples, 2)
    trl_dat = np.vstack([sig1, sig1]).T + noise
    pdiff, freq = rel_phases(trl_dat, taper='dpss', taper_opt={'Kmax': 5, 'NW': 2}, samplerate=fs)

    rel_ph[..., i] = pdiff

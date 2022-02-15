# -*- coding: utf-8 -*-
#
# Load data Field Trip .mat files
#

# Builtin/3rd party package imports
import numpy as np
from scipy import io as sio
import re
import h5py
from tqdm import tqdm

# Local imports
from syncopy.shared.errors import (SPYTypeError, SPYValueError, SPYIOError, SPYInfo, 
                                   SPYError, SPYWarning)

from syncopy.datatype import AnalogData


__all__ = ["load_ft_raw"]


def load_ft_raw(filename,
                select_structures=None,
                add_fields=None,
                mem_use=2000):

    '''
    Imports raw time-series data from Field Trip
    into potentially multiple `~syncopy.AnalogData` objects,
    one for each structure found within the MAT-file.

    For MAT-File < v7.3 the MAT-file gets loaded completely
    into RAM, but its size should be capped at 2GB.
    The v7.3 is in hdf5 format and will be read in trial-by-trial,
    this should be the default for MAT-Files exceeding 2GB.

    The aim is to parse each FT data structure, which
    have the following fields (Syncopy analogon on the right):

    FT     Syncopy

    label - channel
    trial - trial
    time  - time

    optional:
    fsample - samplerate

    cfg - ?

    The FT `cfg` contains a lot of meta data which at the
    moment we don't import into Syncopy.

    This is still experimental code, use with caution!!

    Parameters
    ----------
    filename: str
        Path to the MAT-File
    select_structures: sequence or None, optional
        Sequence of strings, one for each structure,
        the default `None` will load all structures found
    add_fields: sequence, optional
        Additional MAT-File fields within each structure to
        be imported. They can be accessed via the `AnalogData.info` attribute.
    mem_use: int
        The amount of RAM requested for the import process in MB. Note that < v7.3 MAT-File formats can only be loaded at once. For MAT-File v7.3 this should be at least twice the size of a single trial.

    Returns
    -------
    out_dict: dict
        Dictionary with keys being the names of the structures loaded from the MAT-File,
        and as values the `~syncopy.AnalogData` datasets

    See also
    --------
    MAT-File formats: https://de.mathworks.com/help/matlab/import_export/mat-file-versions.html
    Field Trip datastructures: https://www.fieldtriptoolbox.org/development/datastructure/

    Examples
    --------
    Load the two structures from a MAT-File `example.mat`:

    dct = load_ft_raw('example.mat', select_structures=('Data_K', Data_KB'))

    Access the individual `~syncopy.AnalogData` datasets:

    data_kb = dct['Data_KB']
    data_k = dct['Data_K']

    '''

    # Required fields for the ft_datatype_raw
    req_fields_raw = ('time', 'trial', 'label')

    version = _get_Matlab_version(filename)
    msg = f"Reading MAT-File version {version} "
    SPYInfo(msg)

    # new hdf container format, use h5py
    if version >= 7.3:

        h5File = h5py.File(filename, 'r')
        struct_keys = [key for key in h5File.keys() if '#' not in key]

        struct_container = h5File
        struct_reader = lambda struct: _read_hdf_structure(struct,
                                                           h5File=h5File,
                                                           mem_use=mem_use,
                                                           add_fields=add_fields)

    # old format <2GB, use scipy's MAT reader
    else:

        if mem_use < 2000:
            msg = "MAT-File version < 7.3 does not support lazy loading"
            msg += f"\nReading {filename} might take up to 2GB of RAM, you requested only {mem_use / 1000}GB"
            SPYWarning(msg)
            
        raw_dict = sio.loadmat(filename,
                               mat_dtype=True,
                               simplify_cells=True)

        struct_keys = [skey for skey in raw_dict.keys() if '__' not in skey]

        struct_container = raw_dict
        struct_reader = lambda struct: _read_dict_structure(struct,
                                                            add_fields=add_fields)

    if len(struct_keys) == 0:
        SPYValueError(legal="At least one structure",
                      varname=filename,
                      actual="No structure found"
                      )

    msg = f"Found {len(struct_keys)} structure(s): {struct_keys}"
    SPYInfo(msg)

    out_dict = {}

    # load only a subset
    if select_structures is not None:
        keys = select_structures
    # load all structures found
    else:
        keys = struct_keys

    for skey in keys:
        if skey not in struct_keys:
            msg = f"Could not find structure `{skey}` in {filename}"
            SPYWarning(msg)
            continue

        structure = struct_container[skey]
        _check_req_fields(req_fields_raw, structure)
        data = struct_reader(structure)
        out_dict[skey] = data

    return out_dict


def _read_hdf_structure(h5Group,
                        h5File,
                        mem_use,
                        add_fields=None):

    '''
    Each Matlab structure contained in
    a hdf5 MAT-File is a h5py Group object.

    Each key of this Group corresponds to
    a field in the Matlab structure.

    This is the translation from FT to Syncopy:

    FT     Syncopy

    label - channel
    trial - trial
    time  - time

    optional:
    fsample - samplerate

    cfg - X

    '''
    # for user info
    struct_name = h5Group.name[1:]

    # this should be fixed upstream such that
    # the `defaultDimord` is indeed the default :)
    AData = AnalogData(dimord=AnalogData._defaultDimord)

    # probably better to define an abstract mapping
    # if we want to support more FT formats in the future

    # these are numpy arrays holding hdf5 object references
    # e.i. one per trial, channel, time (per trial)
    trl_refs = h5Group['trial'][:, 0]
    time_refs = h5Group['time'][:, 0]
    chan_refs = h5Group['label'][0, :]

    if 'fsample' in h5Group:
        AData.samplerate = h5Group['fsample'][0, 0]
    else:
        AData.samplerate = _infer_fsample(h5File[time_refs[0]])

    # -- retrieve shape information --
    nTrials = trl_refs.size

    # peek in 1st trial to determine single trial shape
    # we only support equal trial lengths at this stage
    nSamples, nChannels = h5File[trl_refs[0]].shape
    nTotalSamples = nTrials * nSamples

    itemsize = h5File[trl_refs[0]].dtype.itemsize
    # in Mbyte
    trl_size = itemsize * nSamples * nChannels / 1e6

    # assumption: single trial fits into RAM
    if trl_size > 0.4 * mem_use:
        msg = f"\nSingle trial is at least 40% of the requested chache size of {mem_use}MB\n"
        msg += f"Still trying to load {trl_size:.1f}MB trials.."
        SPYWarning(msg)

    # -- IO process --

    # create new hdf5 dataset for our AnalogData
    # with the default dimord ['time', 'channel']
    h5FileOut = h5py.File(AData.filename, mode="w")
    ADset = h5FileOut.create_dataset("data",
                                     dtype=np.float32,
                                     shape=[nTotalSamples, nChannels])

    pbar = tqdm(trl_refs, desc=f"{struct_name} - loading {nTrials} trials")
    SampleCounter = 0   # trial stacking

    # one swipe per trial
    for tr in pbar:
        ADset[SampleCounter:nSamples, :] = h5File[tr]
        SampleCounter += nSamples
    pbar.close()

    AData.data = ADset

    # -- trialdefinition --

    nTr_rng = np.arange(nTrials)
    sampleinfo = np.vstack([nTr_rng, nTr_rng + 1]).T * nSamples

    offsets = []
    # we need to look into the time vectors for each trial
    for time_r in time_refs:
        offsets.append(h5File[time_r][0, 0])
    offsets = np.array(offsets)

    trl_def = np.hstack([sampleinfo, offsets[:, None]])
    AData.trialdefinition = trl_def

    # each channel label is an integer array with shape (X, 1),
    # where `X` is the number of ascii encoded characters
    channels = [''.join(map(chr, h5File[cr][:, 0])) for cr in chan_refs]
    AData.channel = channels

    # -- additional fields --

    # this is the most experimental part
    AData.info = {}
    for field in add_fields:
        if field not in h5Group:
            msg = f"Could not find additional field {field} in {struct_name}"
            SPYWarning(msg)
            continue

        # again an array of hdf5 object references
        af_refs = h5Group[field][:, 0]

        AData.info[field] = []
        for af_ref in af_refs:
            # here would be more parsing needed
            # if we want to generally (strings, numbers, ..)
            # support this
            AData.info[field] = h5File[af_ref]

    return AData


def _read_dict_structure(structure, add_fields=None):

    '''
    Local helper to parse a single FT structure
    and return an `~syncopy.AnalogData` object

    Only for for Matlab data format version < 7.3
    which was opened via scipy.io.loadmat!

    This is the translation from FT to Syncopy:

    FT     Syncopy

    label - channel
    trial - trial
    time  - time

    optional:
    fsample - samplerate

    cfg - X

    Each trial in FT has nChannels x nSamples ordering,
    Syncopy has nSamples x nChannels
    '''
    
    # nTrials = structure["trial"].shape[0]
    trials = []
        
    # 1st trial as reference
    nChannels, nSamples = structure['trial'][0].shape

    # check equal trial lengths
    for trl in structure['trial']:

        if trl.shape[-1] != nSamples:
            lgl = 'Trials of equal lengths'
            actual = 'Trials of unequal lengths'
            raise SPYValueError(lgl, varname="load .mat", actual=actual)

        # channel x sample ordering in FT
        trials.append(trl.T.astype(np.float32))

    # initialize AnalogData
    if 'fsample' in structure:
        samplerate = structure['fsample']
    else:
        samplerate = _infer_fsample(structure['time'][0])

    AData = AnalogData(trials, samplerate=samplerate)

    AData.info = {}
    # get the channel ids
    channels = structure['label']
    # set the channel ids
    AData.channel = list(channels.astype(str))

    # update trialdefinition
    times_array = np.vstack(structure['time'])

    # nTrials x nSamples
    offsets = times_array[:, 0] * AData.samplerate

    trl_def = np.hstack([AData.sampleinfo, offsets[:, None]])
    AData.trialdefinition = trl_def

    # write additional fields(non standard FT-format)
    # into Syncopy config
    afields = add_fields if add_fields is not None else range(0)
    for field in afields:
        AData.info[field] = structure[field]
    return AData


def _get_Matlab_version(filename):

    '''
    Peeks into the 1st line of a .mat file
    and extracts the version information.
    Works for both < 7.3 and newer MAT-files.
    '''

    with open(filename, 'rb') as matfile:
        line1 = next(matfile)
        # relevant information
        header = line1[:76].decode()

    # matches for example 'MATLAB 5.01'
    # with the version as only capture group

    pattern = re.compile("^MATLAB\s(\d*\.\d*)")
    match = pattern.match(header)

    if not match:
        lgl = 'recognizable .mat file'
        actual = 'can not recognize .mat file'
        raise SPYValueError(lgl, filename, actual)

    version = float(match.group(1))

    return version


def _check_req_fields(req_fields, structure):

    '''
    Just check the the minimal required fields
    (aka keys in Python) are present in a
    Matlab structure

    Works for both old-style (dict) and
    new-style (hdf5 Group) MAT-file structures.
    '''

    for key in req_fields:
        if key not in structure:
            lgl = f"{key} present in MAT structure"
            actual = f"{key} missing"
            raise SPYValueError(lgl, 'MAT structure', actual)


def _infer_fsample(time_vector):

    '''
    Akin to `ft_datatype_raw` determine
    the sampling frequency from the sampling
    times
    '''

    return np.mean(np.diff(time_vector))

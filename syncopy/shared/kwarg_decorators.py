# -*- coding: utf-8 -*-
# 
# Decorators for Syncopy metafunctions and `computeFunction`s
# 
# Created: 2019-10-22 10:56:32
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-10-24 14:32:33>

# Builtin/3rd party package imports
import functools
import h5py
import inspect
import numpy as np

# Local imports
from syncopy.shared.errors import SPYIOError, SPYTypeError, SPYValueError, SPYError
from syncopy.shared.parsers import get_defaults
import syncopy as spy
if spy.__dask__:
    import dask.distributed as dd

__all__ = []


def unwrap_cfg(func):
    """
    Decorator that unwraps `cfg` "structure" in metafunction call

    Parameters
    ----------
    func : callable
        Typically a Syncopy metafunction such as :func:`~syncopy.freqanalysis`

    Returns
    -------
    wrapper_cfg : callable
        Wrapped function; `wrapper_cfg` extracts keyword arguments from a possibly
        provided `cfg` option "structure" based on the following logic:
        
        1. Probe positional argument list of `func` for regular Python dict or
           :class:`~syncopy.StructDict`. *Every hit* is assumed to be a `cfg` option 
           "structure" and removed from the list. Raises a 
           :class:`~syncopy.shared.errors.SPYValueError` if (a) more than one 
           dict or :class:`~syncopy.StructDict` is found in provided positional 
           arguments (b) keywords are provided in addition to `cfg` (c) `cfg` is 
           provided as positional as well as keyword argument. 
        2. If no `cfg` is found in positional arguments, check `func`'s keyword
           arguments for a provided `cfg` entry. Raises a 
           :class:`~syncopy.shared.errors.SPYTypeError` if `cfg` keyword 
           entry is not a Python dict or :class:`~syncopy.StructDict`. 
        3. If `cfg` was found either in positional or keyword arguments, then 
           (a) process its "linguistic" boolean keys (convert any "yes"/"no" entries 
           to `True` /`False`) and then (b) extract an existing "data" entry and 
           create a `data` variable. Raises a :class:`~syncopy.shared.errors.SPYValueError`
           if `cfg` contains both a "data" and "dataset" entry. 
        4. Perform the actual unwrapping: at this point, a provided `cfg` only 
           contains keyword arguments of `func`. If the (first) input object `data` 
           was provided as `cfg` entry, it already exists in the local namespace. 
           If not, then by convention, it is the first element of the (remaining) 
           positional argument list. Thus, the meta-function can now be called via
           ``func(data, *args, **kwargs)``. 
        5. Amend the docstring of `func`: add a one-liner mentioning the possibility
           of using `cfg` when calling `func` to the header of its docstring. 
           Append a paragraph to the docstrings' "Notes" section illustrating 
           how to call `func` with a `cfg` option "structure" that specifically 
           uses `func` and its input parameters Note: both amendments are only 
           inserted in `func`'s docstring if the respective sections already exist. 
           
    Notes
    -----
    This decorator is primarily intended as bookkeeper for Syncopy metafunctions. 
    It permits "FieldTrip-style" calls of Syncopy metafunctions by "Pythonizing" 
    (processing and subsequent unpacking) of dict-like `cfg` objects. This 
    standardization allows all other Syncopy decorators (refer to See also section) 
    to safely use standard Python ``*args`` and ``**kwargs`` input arguments. 
    
    Supported call signatures:
    
    * ``func(cfg, data)``: `cfg` exclusively contains keyword arguments of `func`, 
      `data` is a Syncopy data object. 
    * ``func(data, cfg)``: same as above
    * ``func(data, cfg=cfg)``: same as above, but `cfg` itself is provided as 
      keyword argument
    * ``func(cfg)``: `cfg` contains a field `data` or `dataset` (not both!) 
      holding a Syncopy data object used as input of `func`
    * ``func(cfg=cfg)``: same as above with `cfg` being provided as keyword
    * ``func(data, kw1=val1, kw2=val2)``: standard Python call style with keywords
      being provided explicitly 
      
    Invalid call signatures:
    
    * ``func(data, cfg, cfg=cfg)``: `cfg` must not be provided as positional and 
      keyword argument
    * ``func(data, cfg, kw1=val1)``: if `cfg` is provided, any non-default
      keyword-values must be provided as `cfg` entries
    * ``func(cfg, {})``: every dict in `func`'s positional argument list is interpreted
      as `cfg` "structure"
    * ``func(data, cfg=value)``: `cfg` must be a Python dict or :class:`~syncopy.StructDict`
    
    See also
    --------
    unwrap_select : extract `select` keyword and process in-place data-selections
    unwrap_io : set up 
                :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`-calls 
                based on parallel processing setup
    detect_parallel_client : controls parallel processing engine via `parallel` keyword
    _append_docstring : local helper for manipulating docstrings
    _append_signature : local helper for manipulating function signatures
    """

    # Perform a little introspection gymnastics to get the name of the first 
    # positional and keyword argument of `func`
    funcParams = inspect.signature(func).parameters
    paramList = list(funcParams)
    kwargList = [pName for pName, pVal in funcParams.items() if pVal.default != pVal.empty]
    arg0 = paramList[0]
    kwarg0 = kwargList[0]

    @functools.wraps(func)
    def wrapper_cfg(*args, **kwargs):
        
        # First, parse positional arguments for dict-type inputs (`k` counts the 
        # no. of dicts provided) and convert tuple of positional args to list
        cfg = None
        k = 0
        args = list(args)
        for argidx, arg in enumerate(args):
            if isinstance(arg, dict):
                cfgidx = argidx
                k += 1

        # If a dict was found, assume it's a `cfg` dict and extract it from
        # the positional argument list; if more than one dict was found, abort
        # IMPORTANT: create a copy of `cfg` using `StructDict` constructor to
        # not manipulate `cfg` in user's namespace!
        if k == 1:
            cfg = spy.StructDict(args.pop(cfgidx))
        elif k > 1:
            raise SPYValueError(legal="single `cfg` input",
                                varname="cfg",
                                actual="{0:d} `cfg` objects in input arguments".format(k))
            
        # Now parse provided keywords for `cfg` entry - if `cfg` was already
        # provided as positional argument, abort
        # IMPORTANT: create a copy of `cfg` using `StructDict` constructor to
        # not manipulate `cfg` in user's namespace!
        if kwargs.get("cfg") is not None:
            if cfg:
                lgl = "`cfg` either as positional or keyword argument, not both"
                raise SPYValueError(legal=lgl, varname="cfg")
            cfg = spy.StructDict(kwargs.pop("cfg"))
            if not isinstance(cfg, dict):
                raise SPYTypeError(kwargs["cfg"], varname="cfg",
                                   expected="dictionary-like")

        # If `cfg` was detected either in positional or keyword arguments, process it
        if cfg:

            # If a method is called using `cfg`, non-default values for
            # keyword arguments *have* to be provided within the `cfg`
            defaults = get_defaults(func)
            for key, value in kwargs.items():
                if defaults[key] != value:
                    raise SPYValueError(legal="no keyword arguments",
                                        varname=key,
                                        actual="non-default value for {}".format(key))

            # Translate any existing "yes" and "no" fields to `True` and `False`,
            # respectively, and subsequently call the function with `cfg` unwrapped
            for key in cfg.keys():
                if str(cfg[key]) == "yes":
                    cfg[key] = True
                elif str(cfg[key]) == "no":
                    cfg[key] = False

            # If `cfg` contains keys 'data' or 'dataset' extract corresponding
            # entry and make it a positional argument (abort if both 'data'
            # and 'dataset' are present)
            data = cfg.pop("data", None)
            if cfg.get("dataset"):
                if data:
                    lgl = "either 'data' or 'dataset' field in `cfg`, not both"
                    raise SPYValueError(legal=lgl, varname="cfg")
                data = cfg.pop("dataset")
            if data:
                args = [data] + args
                
            # Input keywords are all provided by `cfg`
            kwords = cfg
            
        else:
        
            # No meaningful `cfg` keyword found: take standard input keywords
            kwords = kwargs
            
        # Remove data (always first positional argument) from anonymous `args` list
        if len(args) == 0:
            err = "{0} missing mandatory positional argument: `{1}`"
            raise SPYError(err.format(func.__name__, arg0))
        data = args.pop(0)
            
        # Call function with modified positional/keyword arguments
        return func(data, *args, **kwords)

    # Append one-liner to docstring header mentioning the use of `cfg`
    introEntry = \
    "    \n" +\
    "    The parameters listed below can be provided as is or a via a `cfg`\n" +\
    "    configuration 'structure', see Notes for details. \n"
    wrapper_cfg.__doc__ = _append_docstring(wrapper_cfg, 
                                            introEntry, 
                                            insert_in="Header", 
                                            at_end=True)
    
    # Append a paragraph explaining the use of `cfg` by an example that explicitly
    # mentions `func`'s name and input parameters
    notesEntry = \
    "    This function can be either called providing its input arguments directly\n" +\
    "    or via a `cfg` configuration 'structure'. For instance, the following function\n" +\
    "    calls are equivalent\n" +\
    "    \n" +\
    "    >>> spy.{fname:s}({arg0:s}, {kwarg0:s}=...)\n" +\
    "    >>> cfg = spy.StructDict()\n" +\
    "    >>> cfg.{kwarg0:s} = ...\n" +\
    "    >>> spy.{fname:s}(cfg, {arg0:s})\n" +\
    "    >>> cfg.{arg0:s} = {arg0:s}\n" +\
    "    >>> spy.{fname:s}(cfg)\n" +\
    "    \n" +\
    "    Please refer to :doc:`/user/fieldtrip` for further details. \n\n"
    wrapper_cfg.__doc__ = _append_docstring(wrapper_cfg, 
                                            notesEntry.format(fname=func.__name__,
                                                              arg0=arg0,
                                                              kwarg0=kwarg0), 
                                            insert_in="Notes", 
                                            at_end=False)

    return wrapper_cfg


def unwrap_select(func):
    """
    Decorator for handling in-place data selections via `select` keyword
    
    Parameters
    ----------
    func : callable
        Typically a Syncopy metafunction such as :func:`~syncopy.freqanalysis`

    Returns
    -------
    wrapper_select : callable
        Wrapped function; `wrapper_select` extracts `select` from keywords
        provided to `func` and uses it to set the `._selector` property of the 
        input object. After successfully calling `func` with the modified input, 
        `wrapper_select` modifies `func` itself:
        
        1. The "Parameters" section in the docstring of `func` is amended by an 
           entry explaining the usage of `select` (that mostly points to 
           :func:`~syncopy.selectdata`). Note: `func`'s docstring is only extended
           if it has a "Parameters" section. 
        2. If not already present, `select` is added as optional keyword (with 
           default value `None`) to the signature of `func`. 
            
    Notes
    -----
    This decorator assumes that the `func` has already been processed by 
    :func:`~syncopy.shared.kwarg_decorators.unwrap_cfg` and hence expects the call signature 
    of `func` to be of the form ``func(data, *args, **kwargs)``. In other words, 
    :func:`~syncopy.shared.kwarg_decorators.unwrap_select` is intended as "inner" decorator 
    of metafunctions, for instance
    
    .. code-block:: python
    
        @unwrap_cfg
        @unwrap_select
        def somefunction(data, kw1="default", kw2=None, **kwargs):
        ...
        
    **Important** The metafunction `func` **must** accept "anonymous" keywords
    via a ``**kwargs`` dictionary. This requirement is due to the fact that 
    :func:`~syncopy.shared.kwarg_decorators.unwrap_cfg` cowardly refuses to change 
    the byte-code of `func`, that is, `select` is not actually added as a new 
    keyword to `func`, only the corresponding signature is manipulated. 
    Thus, if `func` does not support a `kwargs` parameter dictionary, 
    using this decorator will have *strange* consequences. Specifically, `select` 
    will show up in `func`'s signature but it won't be actually usable:

    .. code-block:: python
    
        @unwrap_cfg
        @unwrap_select
        def somefunction(data, kw1="default", kw2=None):
        ...
        
        >>> help(somefunction)
        somefunction(data, kw1="default", kw2=None, select=None)
        ...
        >>> somefunction(data, select=None)
        TypeError: somefunction() got an unexpected keyword argument 'select' 
        
    
    See also
    --------
    unwrap_cfg : Decorator for processing `cfg` "structs"
    detect_parallel_client : controls parallel processing engine via `parallel` keyword
    """

    @functools.wraps(func)
    def wrapper_select(data, *args, **kwargs):
        
        # Process data selection: if provided, extract `select` from input kws
        data._selection = kwargs.get("select")
        
        # Call function with modified data object
        res = func(data, *args, **kwargs)
        
        # Wipe data-selection slot to not alter user objects
        data._selection = None

        return res
    
    # Append `select` keyword entry to wrapped function's docstring and signature
    selectDocEntry = \
    "    select : dict or :class:`~syncopy.datatype.base_data.StructDict` or str\n" +\
    "        In-place selection of subset of input data for processing. Please refer\n" +\
    "        to :func:`syncopy.selectdata` for further usage details. \n"
    wrapper_select.__doc__ = _append_docstring(func, selectDocEntry)
    wrapper_select.__signature__ = _append_signature(func, "select")
    
    return wrapper_select        


def detect_parallel_client(func):
    """
    Decorator for handling parallelization via `parallel` keyword/client detection
    
    Parameters
    ----------
    func : callable
        Typically a Syncopy metafunction such as :func:`~syncopy.freqanalysis`

    Returns
    -------
    parallel_client_detector : callable
        Wrapped function; `parallel_client_detector` attempts to extract `parallel` 
        from keywords provided to `func`. If `parallel` is `True` or `None` 
        (i.e., was not provided) and dask is available, `parallel_client_detector` 
        attempts to connect to a running dask parallel processing client. If successful, 
        `parallel` is set to `True` and updated in `func`'s keyword argument dict. 
        If no client is found or dask is not available, a corresponding warning 
        message is printed and `parallel` is set to `False` and updated in `func`'s 
        keyword argument dict. After successfully calling `func` with the modified 
        input arguments, `parallel_client_detector` modifies `func` itself:
        
        1. The "Parameters" section in the docstring of `func` is amended by an 
           entry explaining the usage of `parallel`. Note: `func`'s docstring is 
           only extended if it has a "Parameters" section.  
        2. If not already present, `parallel` is added as optional keyword (with 
           default value `None`) to the signature of `func`. 
            
    Notes
    -----
    This decorator assumes that `func` has already been processed by 
    :func:`~syncopy.shared.kwarg_decorators.unwrap_cfg` and hence expects the call 
    signature of `func` to be of the form ``func(data, *args, **kwargs)``. 
    In other words, :func:`~syncopy.shared.kwarg_decorators.detect_parallel_client`
    is intended as "inner" decorator of metafunctions, for instance. See Notes in 
    the docstring of :func:`~syncopy.shared.kwarg_decorators.unwrap_select` for 
    further details. 
    
    See also
    --------
    unwrap_select : extract `select` keyword and process in-place data-selections
    unwrap_cfg : Decorator for processing `cfg` "structs"
    """
    
    @functools.wraps(func)
    def parallel_client_detector(data, *args, **kwargs):

        # Extract `parallel` keyword: if `parallel` is `False`, nothing happens
        parallel = kwargs.get("parallel")
        
        # For later reference: dynamically fetch name of wrapped function
        funcName = "Syncopy <{}>".format(func.__name__)

        # Detect if dask client is running and set `parallel` keyword accordingly
        if parallel is None or parallel is True:
            if spy.__dask__:
                try:
                    dd.get_client()
                    parallel = True
                except ValueError:
                    if parallel is True:
                        wrng = \
                        "{name:s} WARNING: no running parallel processing client " +\
                        "found. Please use `spy.esi_cluster_setup` to launch a " +\
                        "distributed computing cluster. Computation will be performed" +\
                        "sequentially. "
                        print(wrng.format(name=funcName))
                    parallel = False
            else:
                wrng = \
                "{name:s} WARNING: dask seems not to be installed on this system. " +\
                "Parallel processing capabilities cannot be used. "
                print(wrng.format(name=funcName))
                parallel = False
                
        # Add/update `parallel` to/in keyword args
        kwargs["parallel"] = parallel

        return func(data, *args, **kwargs)
    
    # Append `parallel` keyword entry to wrapped function's docstring and signature
    parallelDocEntry = \
    "    parallel : None or bool\n" +\
    "        If `None` (recommended), processing is automatically performed in \n" +\
    "        parallel (i.e., concurrently across trials/channel-groups), provided \n" +\
    "        a dask parallel processing client is running and available. \n" +\
    "        Parallel processing can be manually disabled by setting `parallel` \n" +\
    "        to `False`. If `parallel` is `True` but no parallel processing client\n" +\
    "        is running, computing will be performed sequentially. \n"
    parallel_client_detector.__doc__ = _append_docstring(func, parallelDocEntry)
    parallel_client_detector.__signature__ = _append_signature(func, "parallel")
    
    return parallel_client_detector        


def unwrap_io(func):
    """
    Decorator for handling parallel execution of a
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    
    Parameters
    ----------
    func : callable
        A Syncopy :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
        
    Returns
    -------
    wrapper_io : callable
        Wrapped function; `wrapper_io` changes the way it invokes the wrapped 
        `computeFunction` and processes its output based on the type of the 
        provided first positional argument `trl_dat`. 
        
        * `trl_dat` : dict
          Wrapped `computeFunction` is executed concurrently; `trl_dat` was 
          assembled by 
          :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.compute_parallel`
          and contains information for parallel workers (particularly, paths and 
          dataset indices of HDF5 files for reading source data and writing results). 
          Nothing is returned (the output of the wrapped `computeFunction` is
          directly written to disk). 
        * `trl_dat` : :class:`numpy.ndarray` or :class:`~syncopy.datatype.base_data.FauxTrial` object
          Wrapped `computeFunction` is executed sequentially (either during dry-
          run phase or in purely sequential computations); `trl_dat` is directly
          propagated to the wrapped `computeFunction` and its output is returned
          (either a tuple or :class:`numpy.ndarray`, depending on the value of 
          `noCompute`, see 
          :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
          for details)
          
    Notes
    -----
    Parallel execution supports two writing modes: concurrent storage of results
    in multiple HDF5 files or sequential writing of array blocks in a single 
    output HDF5 file. In both situations, the output array returned by 
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    is immediately written to disk and **not** propagated back to the caller to 
    avoid inter-worker network communication. 
    
    In case of parallel writing, trial-channel blocks are stored in individual 
    HDF5 containers (virtual sources) that are consolidated into a single 
    :class:`h5py.VirtualLayout` which is subsequently used to allocate a virtual 
    dataset inside a newly created HDF5 file (located in Syncopy's temporary 
    storage folder). 

    Conversely, in case of sequential writing, each resulting array is written 
    sequentially to an existing single output HDF5 file using  a distributed mutex 
    for access control to prevent write collisions. 

    See also
    --------
    unwrap_cfg : Decorator for processing `cfg` "structs"
    """

    @functools.wraps(func)
    def wrapper_io(trl_dat, *args, **kwargs):

        # `trl_dat` is a NumPy array or `FauxTrial` object: execute the wrapped 
        # function and return its result
        if not isinstance(trl_dat, dict):
            return func(trl_dat, *args, **kwargs)

        # The fun part: `trl_dat` is a dictionary holding components for parallelization        
        hdr = trl_dat["hdr"]
        keeptrials = trl_dat["keeptrials"]
        infilename = trl_dat["infile"]
        indset = trl_dat["indset"]
        ingrid = trl_dat["ingrid"]
        sigrid = trl_dat["sigrid"]
        fancy = trl_dat["fancy"]
        vdsdir = trl_dat["vdsdir"]
        outfilename = trl_dat["outfile"]
        outdset = trl_dat["outdset"]
        outgrid = trl_dat["outgrid"]

        # === STEP 1 === read data into memory
        # Generic case: data is either a HDF5 dataset or memmap
        if hdr is None:
            try:
                with h5py.File(infilename, mode="r") as h5fin:
                    if fancy:
                        arr = np.array(h5fin[indset][ingrid])[np.ix_(*sigrid)]
                    else:
                        arr = np.array(h5fin[indset][ingrid])
            except OSError:
                try:
                    if fancy:
                        arr = open_memmap(infilename, mode="c")[np.ix_(*ingrid)]
                    else:
                        arr = np.array(open_memmap(infilename, mode="c")[ingrid])
                except:
                    raise SPYIOError(infilename)
            except Exception as exc:
                raise exc
                
        # For VirtualData objects
        else:
            idx = ingrid
            if fancy:
                idx = np.ix_(*ingrid)
            dsets = []
            for fk, fname in enumerate(infilename):
                dsets.append(np.memmap(fname, offset=int(hdr[fk]["length"]),
                                        mode="r", dtype=hdr[fk]["dtype"],
                                        shape=(hdr[fk]["M"], hdr[fk]["N"]))[idx])
            arr = np.vstack(dsets)

        # === STEP 2 === perform computation
        # Now, actually call wrapped function
        res = func(arr, *args, **kwargs)
        
        # === STEP 3 === write result to disk
        # Write result to stand-alone HDF file or use a mutex to write to a 
        # single container (sequentially)
        if vdsdir is not None:
            with h5py.File(outfilename, "w") as h5fout:
                h5fout.create_dataset(outdset, data=res)
                h5fout.flush()
        else:
            
            # Create distributed lock (use unique name so it's synced across workers)
            lock = dd.lock.Lock(name='sequential_write')

            # Either (continue to) compute average or write current chunk
            lock.acquire()
            with h5py.File(outfilename, "r+") as h5fout:
                target = h5fout[outdset]
                if keeptrials:
                    target[outgrid] = res    
                else:
                    target[()] = np.nansum([target, res], axis=0)
                h5fout.flush()
            lock.release()
                
        return None # result has already been written to disk
        
    return wrapper_io


def _append_docstring(func, supplement, insert_in="Parameters", at_end=True):
    """
    Local helper to automate text insertions in docstrings
    
    Parameters
    ----------
    func : callable
        Typically a (wrapped) Syncopy metafunction such as :func:`~syncopy.freqanalysis`
    supplement : str
        Text entry to be added to `func`'s docstring. Has to be already formatted 
        correctly for its intended destination section, specifically respecting 
        indentation and line-breaks (e.g., following double-indentation of variable 
        descriptions in the "Parameters" section)
    insert_in : str
        Name of section `supplement` should be inserted into. Available options 
        are `"Header"` (part of the docstring before "Parameters"), `"Parameters"`,
        `"Returns"`, `"Notes"` and `"See also"`. Note that the section specified
        by `insert_in` has to already exist in `func`'s docstring, otherwise 
        `supplement` is *not* inserted. 
    at_end : bool
        If `True`, `supplement` is appended at the end of the section specified 
        by `insert_in`. If `False`, `supplement` is included at the beginning of
        the respective section. 
    
    Returns
    -------
    newDocString : str
        A copy of `func`'s docstring with `supplement` inserted at the location
        specified by `insert_in` and `at_end`. 
        
    Notes
    -----
    This routine is a local auxiliary method that is purely intended for internal
    use. Thus, no error checking is performed. 
        
    See also
    --------
    _append_signature : extend a function's signature
    """
    
    if func.__doc__ is None:
        return
    
    # "Header" insertions always work (an empty docstring is enough to do this). 
    # Otherwise ensure the provided `insert_in` section already exists, i.e., 
    # partitioned `sectionHeading` == queried `sectionTitle`
    if insert_in == "Header":
        sectionText, sectionDivider, rest = func.__doc__.partition("Parameters\n")
        textBefore = ""
        sectionHeading = ""
    else:
        sectionTitle = insert_in + "\n"
        textBefore, sectionHeading, textAfter = func.__doc__.partition(sectionTitle)
        if sectionHeading != sectionTitle:  # `insert_in` was not found in docstring
            return func.__doc__
        sectionText, sectionDivider, rest = textAfter.partition("\n\n")
    sectionText = sectionText.splitlines(keepends=True)
    
    if at_end:
        insertAtLine = -1
        while sectionText[insertAtLine].isspace():
            insertAtLine -= 1
        insertAtLine = min(-1, insertAtLine + 1)
    else:
        insertAtLine = 1
    sectionText = "".join(sectionText[:insertAtLine]) +\
                  supplement +\
                  "".join(sectionText[insertAtLine:])

    newDocString = textBefore +\
                   sectionHeading +\
                   sectionText +\
                   sectionDivider +\
                   rest

    return newDocString


def _append_signature(func, kwname, kwdefault=None):
    """
    Local helper to automate keyword argument insertions in function signatures
    
    Parameters
    ----------
    func : callable
        Typically a (wrapped) Syncopy metafunction such as :func:`~syncopy.freqanalysis`
    kwname : str
        Name of keyword argument to be added to `func`'s signature
    kwdefault : None or any valid type
        Default value of keyword argument specified by `kwname`
        
    Returns
    -------
    newSignature : inspect.Signature
        A copy of `func`'s signature with ``kwname=kwdefault`` included as last 
        named keyword argument (before ``**kwargs``). If `kwname` already exists
        in `func`'s named keyword arguments, `newSignature` is an identical copy
        of `func`'s signature. 
        
    Notes
    -----
    This function **does not** change `func`'s byte-code, that is, it does not 
    actually add a new keyword argument to `func` but just appends a named parameter
    to `func`'s signature. As a consequence, `func` **must** accept "anonymous"
    keywords via a ``**kwargs`` dictionary for this manipulation to work as 
    intended. If `func` does not support a ``kwargs`` parameter dictionary, 
    `kwname` with default value `kwdefault` will be listed in `func`'s signature 
    but trying to use it will trigger a "unexpected keyword argument"-`TypeError`. 
    
    This routine is a local auxiliary method that is purely intended for internal
    use. Thus, no error checking is performed. 

    See also
    --------
    _append_docstring : extend a function's signature
    """
    
    funcSignature = inspect.signature(func)
    if kwname in list(funcSignature.parameters):
        newSignature = funcSignature
    else:
        paramList = list(funcSignature.parameters.values())
        keyword = inspect.Parameter(kwname, inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                    default=kwdefault)
        if paramList[-1].name == "kwargs":
            paramList.insert(-1, keyword)
        else:
            paramList.append(keyword)
        newSignature = inspect.Signature(parameters=paramList)
    return newSignature

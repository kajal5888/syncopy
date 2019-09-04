# -*- coding: utf-8 -*-
#
# Test proper functionality of SyNCoPy DiscreteData-type classes
#
# Created: 2019-03-21 15:44:03
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-27 15:03:47>

import os
import tempfile
import time
import pytest
import numpy as np
from syncopy.datatype import AnalogData, SpikeData, EventData
from syncopy.io import save, load
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import construct_spy_filename


class TestSpikeData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    nd = 50
    seed = np.random.RandomState(13)
    data = np.vstack([seed.choice(ns, size=nd),
                      seed.choice(nc, size=nd),
                      seed.choice(int(nc / 2), size=nd)]).T
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_chn = data[:, 1].max() + 1
    num_unt = data[:, 2].max() + 1

    def test_empty(self):
        dummy = SpikeData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == ["sample", "channel", "unit"]
        for attr in ["channel", "data", "sampleinfo", "samplerate",
                     "trialid", "trialinfo", "unit"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpikeData({})

    def test_nparray(self):
        dummy = SpikeData(self.data)
        assert dummy.channel.size == self.num_chn
        assert dummy.sample.size == self.num_smp
        assert dummy.unit.size == self.num_unt
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            SpikeData(np.ones((3,)))

    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = SpikeData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = SpikeData(self.data2, trialdefinition=self.trl,
                          dimord=["unit", "channel", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["channel", "data", "dimord", "sampleinfo",
                         "samplerate", "trialinfo", "unit"]
            dummy = SpikeData(self.data, samplerate=10)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            dummy2 = SpikeData(filename)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy2, dummy3, dummy4  # avoid PermissionError in Windows
            time.sleep(0.1)  # wait to kick-off garbage collection

            # overwrite existing container w/new data
            dummy.samplerate = 20
            dummy.save()
            dummy2 = SpikeData(filename=filename)
            assert dummy2.samplerate == 20
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection
            
            # ensure trialdefinition is saved and loaded correctly
            dummy = SpikeData(self.data, trialdefinition=self.trl, samplerate=10)
            dummy.save(fname, overwrite=True)
            dummy2 = SpikeData(filename)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection

            # swap dimensions and ensure `dimord` is preserved
            dummy = SpikeData(self.data, dimord=["unit", "channel", "sample"], samplerate=10)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = SpikeData(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.unit.size == self.num_smp  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2
            time.sleep(0.1)


class TestEventData():

    # Allocate test-datasets
    nc = 10
    ns = 30
    data = np.vstack([np.arange(0, ns, 5),
                      np.zeros((int(ns / 5), ))]).T
    data[1::2, 1] = 1
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_evt = np.unique(data[:, 1]).size

    adata = np.arange(1, nc * ns + 1).reshape(ns, nc)

    def test_empty(self):
        dummy = EventData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == ["sample", "eventid"]
        for attr in ["data", "sampleinfo", "samplerate", "trialid", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            EventData({})

    def test_nparray(self):
        dummy = EventData(self.data)
        assert dummy.eventid.size == self.num_evt
        assert dummy.sample.size == self.num_smp
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            EventData(np.ones((3,)))

    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = EventData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = EventData(self.data2, trialdefinition=self.trl,
                          dimord=["eventid", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["data", "dimord", "sampleinfo", "samplerate", "trialinfo"]
            dummy = EventData(self.data, samplerate=10)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            dummy2 = EventData(filename)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy2, dummy3, dummy4  # avoid PermissionError in Windows

            # overwrite existing file w/new data
            dummy.samplerate = 20
            dummy.save()
            dummy2 = EventData(filename=filename)
            assert dummy2.samplerate == 20
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection

            # ensure trialdefinition is saved and loaded correctly
            dummy = EventData(self.data, trialdefinition=self.trl, samplerate=10)
            dummy.save(fname, overwrite=True)
            dummy2 = EventData(filename)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)
            del dummy, dummy2

            # swap dimensions and ensure `dimord` is preserved
            dummy = EventData(self.data, dimord=["eventid", "sample"], samplerate=10)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = EventData(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.eventid.size == self.num_smp # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2
            time.sleep(0.1)

    def test_trialsetting(self):

        # Create sampleinfo w/ EventData vs. AnalogData samplerate
        sr_e = 2
        sr_a = 1
        pre = 2
        post = 1
        msk = self.data[:, 1] == 1
        sinfo = np.vstack([self.data[msk, 0] / sr_e - pre,
                           self.data[msk, 0] / sr_e + post]).T
        sinfo_e = np.round(sinfo * sr_e).astype(int)
        sinfo_a = np.round(sinfo * sr_a).astype(int)

        # Compute sampleinfo w/pre, post and trigger
        evt_dummy = EventData(self.data, samplerate=sr_e, mode="r")
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        assert np.array_equal(evt_dummy.sampleinfo, sinfo_e)

        # Compute sampleinfo w/ start/stop combination
        evt_dummy = EventData(self.data, samplerate=sr_e)
        evt_dummy.definetrial(start=0, stop=1)
        sinfo2 = np.vstack([self.data[np.where(self.data[:, 1] == 0)[0], 0],
                            self.data[np.where(self.data[:, 1] == 1)[0], 0]]).T
        assert np.array_equal(sinfo2, evt_dummy.sampleinfo)

        # Same w/ more complicated data array
        samples = np.arange(0, int(self.ns / 3), 3)[1:]
        dappend = np.vstack([samples, np.full(samples.shape, 2)]).T
        data3 = np.vstack([self.data, dappend])
        idx = np.argsort(data3[:, 0])
        data3 = data3[idx, :]
        evt_dummy = EventData(data3, samplerate=sr_e)
        evt_dummy.definetrial(start=0, stop=1)
        assert np.array_equal(sinfo2, evt_dummy.sampleinfo)

        # Compute sampleinfo w/start/stop arrays instead of scalars
        starts = [2, 2, 1]
        stops = [1, 2, 0]
        sinfo3 = np.empty((3, 2))
        dsamps = list(data3[:, 0])
        dcodes = list(data3[:, 1])
        for sk, (start, stop) in enumerate(zip(starts, stops)):
            idx = dcodes.index(start)
            start = dsamps[idx]
            dcodes = dcodes[idx + 1:]
            dsamps = dsamps[idx + 1:]
            idx = dcodes.index(stop)
            stop = dsamps[idx]
            dcodes = dcodes[idx + 1:]
            dsamps = dsamps[idx + 1:]
            sinfo3[sk, :] = [start, stop]
        evt_dummy = EventData(data3, samplerate=sr_e)
        evt_dummy.definetrial(start=[2, 2, 1], stop=[1, 2, 0])
        assert np.array_equal(evt_dummy.sampleinfo, sinfo3)

        # Attach computed sampleinfo to AnalogData (data and data3 must yield identical resutls)
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Compute and attach sampleinfo on the fly
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Extend data and provoke an exception due to out of bounds erro
        smp = np.vstack([np.arange(self.ns, int(2.5 * self.ns), 5),
                         np.zeros((int((1.5 * self.ns) / 5),))]).T
        smp[1::2, 1] = 1
        data4 = np.vstack([data3, smp])
        evt_dummy = EventData(data=data4, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        # with pytest.raises(SPYValueError):
            # ang_dummy.definetrial(evt_dummy)

        # Trimming edges produces zero-length trial
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, clip_edges=True)

        # We need `clip_edges` to make trial-definition work
        data4 = data4[:-2, :]
        data4[-2, 0] = data4[-1, 0]
        evt_dummy = EventData(data=data4, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        # with pytest.raises(SPYValueError):
            # ang_dummy.definetrial(evt_dummy)
        ang_dummy.definetrial(evt_dummy, clip_edges=True)
        assert ang_dummy.sampleinfo[-1, 1] == self.ns

        # Check both pre/start and/or post/stop being None
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(trigger=1, post=post)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, trigger=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(start=0)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(stop=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(trigger=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post)

        # Try to define trials w/o samplerate set
        evt_dummy = EventData(data=self.data)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata)
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)

        # Try to define trials w/o data
        evt_dummy = EventData(samplerate=sr_e)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)

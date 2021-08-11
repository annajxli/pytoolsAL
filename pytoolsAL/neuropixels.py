#!/usr/bin/env python3

import numpy as np
import warnings
import pathlib
import os


class RainierData:
    def __init__(self, datadir, mn, td, en, probe=None, label=None):
        if isinstance(datadir, str):
            datadir = pathlib.Path(datadir)

        self.sessdir = None
        self.datadir = datadir
        self.probedir = None

        self.mn = mn
        self.td = td
        self.en = en
        self.probe = probe
        self.label = label

        self.flipper_tstamps_sync = None
        self.mean_offset = None
        self.spkrate = None

        self.spikes = None
        self._get_session_dir()

        if self.probe is not None:
            self._get_probe_dir()

        self.npys = {}

    def _get_session_dir(self):
        """
        Find dir corresponding to this mouse, date, experiment
        """
        sessdir = self.datadir / self.mn / self.td / self.en
        if not os.path.isdir(sessdir):
            raise FileNotFoundError(f'{sessdir} not found. Exists?')
        else:
            self.sessdir = sessdir

    def _get_probe_dir(self):
        """
        Get full subdir corresponding to neuropixels files
        Currently not able to handle multiple probes, I honestly don't even
        know how that looks in the file structure.
        """
        probedir = []
        for i in os.listdir(self.sessdir):
            # look for all folders starting with probename
            if i.startswith(self.probe):
                if os.path.isdir(self.sessdir / i):
                    probedir.append(i)
        # check that only one folder exists (raise error if mult or none)
        if len(probedir) == 0:
            raise FileNotFoundError(f'No folder found for {self.probe}')
        elif len(probedir) > 1:
            raise ValueError(f'More than one {self.probe} folder found.'
                             + ' Not equipped to handle.')
        else:
            # if just one (expected), set path variables
            p_subs = os.listdir(self.sessdir / probedir[0])
            # p_sub = p_subs[probenum]
            self.probedir = self.sessdir / probedir[0]

    def load_npy_files(self, dtype, flist):
        dtypes_allow = ['ephys', 'wf', 'sync']
        if dtype not in dtypes_allow:
            return ValueError(f'Data type {dtype} not found. Available types: '
                              + f'{dtypes_allow}')
        if dtype is 'ephys':
            for f in flist:
                self.npys[f] = np.load(self.probedir / f'{f}.npy')
        if dtype in ['sync', 'wf']:
            for f in flist:
                self.npys[f] = np.load(self.sessdir / f'{f}.npy')

    def find_flipper_offset(self):
        npys = self.npys
        sync1 = npys[f'{self.probe}_sync']
        sync2 = npys['tl_sync']

        offsets = sync1 - sync2
        std = np.std(offsets)

        if std > 1e-3:
            warnings.warn(f'flipper offsets standard deviation is {std:4f}')

        mean_offset = np.mean(offsets)
        self.mean_offset = mean_offset

    def sync_timestamps(self, timestamps2=None):
        """
        Use sync1 to identify offset from sync2 and adjust timestamps2.
        Default: usually want to sync imaging to probe, so sync1 is p_sync,
        sync2 is tl_sync, timestamps2 is svd timestamps
        Args:
            sync1: array of flipper timestamps
            sync2: array of flipper timestamps
            timestamps2: timestamps to adjust

        Returns:
            synced: timestamps2 adjusted to align with timestamps1
        """
        self.find_flipper_offset()
        mean_offset = self.mean_offset
        npys = self.npys
        if sync1 is None:
            warnings.warn(f'no args passed to sync_timestamps - defaulting to sync imaging to probe')
            timestamps2 = npys['corr/svdTemporalComponents_corr.Timestamps']

        synced = (timestamps2 + mean_offset).reshape(-1)

        self.flipper_tstamps_sync = synced

    def separate_spikes(self):
        """
        Given equal arrays of [spiketimes] and [spikeclusters]
        Find which spike times belong to each unique neuron (cluster)

        Uses spike_times and spike_clusters from loaded npyps
        Assigns separated spikes to self.neurons and self.spikes
        """
        spk_time = self.npys['spike_times']
        spk_clus = self.npys['spike_clusters']

        if len(spk_time) != len(spk_clus):
            raise ValueError('spike times and clusters have uneven lengths. '
                             + 'this should never happen ????')

        neurons = np.unique(spk_clus)
        spks = []
        for neur in neurons:
            ix = np.where(spk_clus == neur)
            times = spk_time[ix]
            spks.append(times)
        spks = np.asarray(spks, dtype='object')

        self.neurons = neurons
        self.spikes = spks / 3e4  # convert to s: remind myself why this is 3e4

    def bin_spikes(self, bins, spks=None, set_self_matrix=False):
        """
        yep

        Args:
            bins: desired bin edges
            spks: (optional) external spike times, e.g. if shuffled
            set_self_matrix: (optional) whether or not to set internal spkmat argument
                e.g. False when using this for spike rate
                if False, returns matrix. if True, returns None.

        Returns:
            spks_mat: array in shape [neurons, bins]
        """
        if spks is None:
            self.separate_spikes()
            spks = self.spikes

        spk_mat = np.zeros((np.max(self.neurons) + 1, len(bins) - 1))
        # spk_mat[:] = np.nan

        for iN, neur in enumerate(spks):
            hist, edges = np.histogram(neur, bins, density=False)
            neur_num = self.neurons[iN]
            spk_mat[neur_num] = hist

        if set_self_matrix:
            self.spk_mat = spk_mat
        else:
            return spk_mat

    def get_spike_rate(self):
        """
        yep
        """
        end_time = np.ceil(np.max(self.npys['spike_times'] / 3e4))
        sec_bins = np.arange(0, end_time, 1)
        spkrate = np.mean(self.bin_spikes(sec_bins, set_self_matrix=False), axis=1)

        self.spkrate = spkrate

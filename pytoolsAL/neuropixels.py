#!/usr/bin/env python3

import numpy as np
import warnings
import pathlib
import os


class RainierData:
    def __init__(self, datadir, mn, td, probe=None):
        if isinstance(datadir, str):
            datadir = pathlib.Path(datadir)
        self.datadir = datadir
        self.mn = mn
        self.td = td
        self.probe = probe

        self._get_session_dir()

        if self.probe is not None:
            self._get_probe_dir()

        self.npys = {}

    def _get_session_dir(self):
        """
        Find dir corresponding to this mouse and date
        Explicitly passing the 1 folder for now. Might want to change later.
        """
        sessdir = self.datadir / self.mn / self.td / '1'
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
                if os.path.isdir(self.sessdir/i):
                    probedir.append(i)
        # check that only one folder exists (raise error if mult or none)
        if len(probedir) == 0:
            raise FileNotFoundError(f'No folder found for {self.probe}')
        elif len(probedir) > 1:
            raise ValueError(f'More than one {self.probe} folder found.'
                             + ' Not equipped to handle.')
        else:
            # if just one (expected), set path variables
            p_sub = os.listdir(self.sessdir/probedir[0])
            self.probedir = self.sessdir/probedir[0]/p_sub[0]

    def load_npy_files(self, dtype, flist):
        dtypes_allow = ['ephys', 'wf', 'sync']
        if dtype not in dtypes_allow:
            return ValueError(f'Data type {dtype} not found. Available types: '
                              + f'{dtypes_allow}')
        if dtype is 'ephys':
            for f in flist:
                self.npys[f] = np.load(self.probedir/f'{f}.npy')
        if dtype in ['sync', 'wf']:
            for f in flist:
                self.npys[f] = np.load(self.sessdir/f'{f}.npy')

    def sync_timestamps(self, sync1=None, sync2=None, timestamps2=None):
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
        npys = self.npys
        if sync1 is None:
            warnings.warn(f'no args passed to sync_timestamps - defaulting to sync imaging to probe')
            sync1 = npys[f'{self.probe}_sync']
            sync2 = npys['tl_sync']
            timestamps2 = npys['corr/svdTemporalComponents_corr.Timestamps']

        offsets = sync1 - sync2
        std = np.std(offsets)

        if std > 1e-3:
            warnings.warn(f'flipper offsets standard deviation is {std:4f}')

        mean_offset = np.mean(offsets)
        synced = (timestamps2 + mean_offset).reshape(-1)

        self.flipper_tstamps_sync = synced
        self.mean_offset = mean_offset

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
        self.spikes = spks/3e4  # convert to s: remind myself why this is 3e4

    def bin_spikes(self, bins, set_self_matrix=False):
        """
        yep

        Args:
            bins: desired bin edges
            set_self_matrix: whether or not to set internal spkmat argument
                e.g. False when using this for spike rate
                if False, returns matrix. if True, returns None.

        Returns:
            spks_mat: array in shape [neurons, bins]
        """
        self.separate_spikes()
        spks = self.spikes
        spks_binned = []
        for neur in spks:
            hist, edges = np.histogram(neur, bins, density=False)
            spks_binned.append(hist)
        spk_mat = np.asarray(spks_binned)

        if set_self_matrix:
            self.spk_mat = spk_mat
        else:
            return spk_mat

    def get_spike_rate(self):
        """
        yep
        """
        tstamps = self.flipper_tstamps_sync
        sec_bins = np.arange(np.min(tstamps), np.max(tstamps), 1)
        spkrate = np.mean(self.bin_spikes(sec_bins, set_self_matrix=False),
                          axis=1)

        self.spkrate = spkrate

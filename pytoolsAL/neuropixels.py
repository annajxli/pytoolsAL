#!/usr/bin/env python3

import numpy as np
import os
import pathlib
from scipy.optimize import curve_fit
import sys
import warnings

sys.path.append('C://Users/anna/Repositories/Pipelines/ephys/qc')
import single_units_wrapper as qc

class RainierData:
    def __init__(self, datadir, mn, td, en, probe=None, label=None):
        if isinstance(datadir, str):
            datadir = pathlib.Path(datadir)

        self.sessdir = None
        self.datadir = datadir
        self.probedir = None

        self.mn = mn
        self.td = td
        self.en = str(en)
        self.probe = probe
        self.label = label

        self.flipper_tstamps_sync = None
        self.mean_offset = None
        self.spk_rate = None

        self.neurons = None
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
        if dtype == 'ephys':
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

    def sync_timestamps(self, sync1=None, timestamps2=None):
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

    def run_refrac_qc(self, bin_size=0.25, thresh=0.1,
                      accept_thresh=0.1, drop=True):
        """
        run the slidingRP_viol qc function on all neurons
        if drop, then drop the failed ones from spk_mat, neurons, spikes, spk_rate
        otherwise, return the boolean accept/reject array
        """
        rp_pass = []
        for spikes in self.spikes:
            accept = qc.slidingRP_viol(spikes)
            rp_pass.append(accept)

        rp_pass = np.array(rp_pass, dtype=bool)
        print(f'{np.sum(rp_pass)}/{len(self.neurons)} neurons passed')

        if drop:
            self.neurons = self.neurons[rp_pass]
            self.spikes = self.spikes[rp_pass]

            self.spk_mat = self.spk_mat[self.neurons]
            self.spk_rate = self.spk_rate[self.neurons]

        else:
            return rp_pass

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
        self.spikes = spks / 3e4  # sample rate, may want to set this
        # programmatically in future

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

        # if there are spikes outside of the supplied bins,
        # drop said spikes
        spks_clipped = []
        for spk_row in spks:
            spk_row = spk_row[spk_row < bins[-1]]
            spks_clipped.append(spk_row)

        spk_mat = np.zeros((np.max(self.neurons) + 1, len(bins) - 1))
        # spk_mat[:] = np.nan

        for iN, neur in enumerate(spks_clipped):
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
        spk_rate = np.mean(self.bin_spikes(sec_bins, set_self_matrix=False), axis=1)

        self.spk_rate = spk_rate


def find_max_channels(templates):
    """
    Given templates.npy file, find the channel with the highest template
    (i.e. neuron) signal (defined by max - min of timesamples).
    Return neuron-length list of channel for each
    Args:
        templates: loaded templates.npy file

    Returns:

    """
    channel_distrib = np.max(templates, axis=1) - np.min(templates, axis=1)
    max_channels = np.argmax(channel_distrib, axis=1)
    return max_channels


def bin_neurons_positions(max_chans, xbins, ybins):
    """

    Args:
        max_chans: array with each neuron's maximum channel position
        xbins: list of x bin edges to use (shanks)
        ybins: list of y bin edges to use (depths)

    Returns:
        bins_dict: dict with key for each combination of [x, y] bin,
            values are the neuron ixs that belong to that bin

    """
    # determine which x and y bins each neuron belongs to
    x_ix = np.digitize(max_chans[:, 0], xbins)
    y_ix = np.digitize(max_chans[:, 1], ybins)

    # make lists of neurons in each pair of x/y bins
    bins_dict = {}
    for yb in range(1, len(ybins)+1):
        for xb in range(1, len(xbins)+1):
            neurs = np.intersect1d(np.argwhere(x_ix == xb), np.argwhere(y_ix == yb))
            binkey = f'{xbins[xb-1]}, {ybins[yb-1]}'
            bins_dict[binkey] = neurs
    return bins_dict


def computePR(spkcounts):
    """
    From Stefano
    """
    C = np.cov(spkcounts)
    ofdiag = ~np.eye(C.shape[0],dtype=bool)
    ondiag = np.eye(C.shape[0],dtype=bool)
    mii = C[ondiag].mean()
    mij = C[ofdiag].mean()
    N = C.shape[0]
    sii = 1 / (N - 1) * np.sum((C[ondiag] - mii) ** 2)
    sij = 1 / (N * (N - 1) - 2) * np.sum((C[ofdiag] - mij) ** 2)
    sijt = np.nan
    siit = np.nan

    N_rep = 1000
    pcg = 0.8
    T = spkcounts.shape[1]
    N = spkcounts.shape[0]

    siiall = np.zeros(N_rep)
    sijall = np.zeros(N_rep)
    Tall = np.zeros(N_rep)
    Nall = np.zeros(N_rep)
    for i_rep in np.arange(N_rep):
        T_num = int(np.random.choice(np.round(np.arange(15, T) * pcg), replace = False))
        N_num = N
        Tall[i_rep] = T_num
        Nall[i_rep] = N_num
        idxs_T = np.random.choice(np.arange(T), T_num, replace = False)
        idxs_N = np.random.choice(np.arange(N), N_num, replace = False)
        X = spkcounts[np.ix_(idxs_N, idxs_T)]
        C = np.cov(X)
        ondiag = np.eye(C.shape[0], dtype=bool)
        ofdiag = np.triu(np.ones_like(C, dtype=bool), k=1)
        sii_temp = C[ondiag].var()
        sij_temp = C[ofdiag].var()
        sijall[i_rep] = sij_temp
        siiall[i_rep] = sii_temp

    def func_cros(x, a, b):
        N_trials = x# N_trials, N_neurons = x
        return (N_trials-1)/N_trials * b + a/N_trials #+ c / (N_trials * (N_neurons+1))

    def func_auto(x, a, b):
        N_trials = x
        return (N_trials - 1) / (N_trials + 1) * b + a / (N_trials + 1)

    sij_fit, pcov = curve_fit(func_cros, Tall, sijall, bounds=(0., [1000., 1000.]))
    sii_fit, pcov = curve_fit(func_auto, Tall, siiall, bounds=(0., [1000., 1000.]))
    sij = sij_fit[1]
    sii = sii_fit[1]

    ds = np.sqrt(sij) / mii
    N = C.shape[0]
    PR = (np.trace(C)) ** 2 / (np.trace(C @ C))
    PRsij = N / (1 + N * ds ** 2)
    PRsijt = ds ** (-2)

    stats = {
    'mii':mii, 'sii':sii, 'mij':mij, 'sij':sij, 'ds':ds, 'N':N,
    'PR':PR, 'PRsij':PRsij, 'PRsijt':PRsijt, 'T':T}

    return stats

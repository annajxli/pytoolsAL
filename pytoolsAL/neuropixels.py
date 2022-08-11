#!/usr/bin/env python3

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pytoolsAL as ptAL
import scipy
from scipy.optimize import curve_fit
import shutil
import sys
import warnings

sys.path.append('C://Users/anna/Repositories/Pipelines/ephys/qc')
# import single_units_wrapper as qc
sys.path.append('C://Users/anna/Repositories/slidingRefractory/python')
import slidingRP as rpqc

def copy_server_data(server_dir, local_dir, mn, file_lim_bytes=1e9, verbose=0):
    """
    copies files below file_lim bytes in a specific subject directory
    skips files if they already exist

    Args:
        serverdir: path to \Subjects folder on server
        localdir: path to equivalent dir on local machine
        mn: mousename (will copy these folders)
        file_lim_bytes: only copy files under this (in bytes), default is
            1e9 bytes (1gb)
        verbose: controls verbosity of output. verbose > 0 displays status
            for each folder and num skipped. verbose > 1 also displays
            skipped files and their sizes.
    """
    subj_dir = Path(server_dir)/mn
    local_subj_dir = Path(local_dir)/mn

    for root, subdirs, files in os.walk(subj_dir):
        for subdir in subdirs:
            # first make all subdirectories (e.g. date, exp)
            old_subdir = Path(root)/subdir
            new_subdir = str(old_subdir).replace(
                str(subj_dir), str(local_subj_dir))
            Path(new_subdir).mkdir(parents=True, exist_ok=True)

        dsp = display(display_id=True)
        skipped = 0
        for iF, file in enumerate(files):
            if verbose > 0:
                dsp.update(f'copying {iF+1}/{len(files)} in {root}')
            old_filepath = Path(root)/file
            filesize = os.path.getsize(old_filepath)

            if filesize < file_lim_bytes:
                new_filepath = str(old_filepath).replace(
                    str(subj_dir), str(local_subj_dir))

                if not Path(new_filepath).is_file():
                    shutil.copy(old_filepath, new_filepath)
                else:
                    print(f'    did not copy {new_filepath} - already exists')

            else:
                skipped += 1
                if verbose > 1:
                    print(f'    skipping {file}: {filesize/1e9:.0f} gb')
        if verbose > 0:
            print(f'    skipped {skipped} oversize files')


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

        self.rp_pass = None
        self.mincont = None

        self.spk_mat = None
        self.positions = None
        self.spk_mat_norm = None

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
            if len(p_subs) == 1:
                probedir[0] = Path(probedir[0])/p_subs[0]

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
        if dtype == 'sync':
            for f in flist:
                self.npys[f] = np.load(self.sessdir / f'{f}.npy')
        if dtype == 'wf':
            for f in flist:
                try:
                    self.npys[f] = np.load(self.sessdir / 'corr' / f'{f}.npy')
                except FileNotFoundError:
                    try:
                        self.npys[f] = np.load(self.sessdir / 'blue' / f'{f}.npy')
                    except FileNotFoundError:
                        raise FileNotFoundError(f'file {f} not found in blue nor in corr')

    def find_flipper_offset(self):
        """
        """
        warnings.warn('this is deprecated, do we still need it?')
        # npys = self.npys
        # sync1 = npys[f'{self.probe}_sync']
        # sync2 = npys['tl_sync']
        #
        # offsets = sync1 - sync2
        # std = np.std(offsets)
        #
        # if std > 1e-3:
        #     warnings.warn(f'flipper offsets standard deviation is {std:4f}')
        #
        # mean_offset = np.mean(offsets)
        # self.mean_offset = mean_offset

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
        warnings.warn('this is deprecated, do we still need it?')
        # self.find_flipper_offset()
        # mean_offset = self.mean_offset
        # npys = self.npys
        # if sync1 is None:
        #     warnings.warn(f'no args passed to sync_timestamps - \
        #                   defaulting to sync imaging to probe')
        #     timestamps2 = npys['corr/svdTemporalComponents_corr.Timestamps']
        #
        # synced = (timestamps2 + mean_offset).reshape(-1)
        #
        # self.flipper_tstamps_sync = synced

    def get_positions(self):
        chan_pos = self.npys['channel_positions']
        max_channels = _find_max_channels(self.npys['templates'])
        neuron_pos = chan_pos[max_channels]

        if self.rp_pass is not None:
            neuron_pos = neuron_pos[self.rp_pass]

        self.positions = neuron_pos

    def run_refrac_qc(self, threshold=10, verbose=0, doPlot=True, drop=True):
        """
        run the slidingRP_viol qc function on all neurons
        if drop, then drop the failed ones from spk_mat, neurons, spikes, spk_rate
        otherwise, return the boolean accept/reject array

        Args:
            threshold: cutoff for minimum contamination at 90% confidence
            verbose: spits out progress
            doPlot: visualize hist of all neurons
            drop: whether to drop failed neurons
        """
        try:
            rp_mincont = np.load(self.probedir / 'min_cont.npy')
            print('qc save file found')

        except:
            print('first time running qc, takes a minute')
            dsp = display(display_id=True)

            rp_mincont = []
            spk_time = self.npys['spike_times']
            spk_clus = self.npys['spike_clusters']

            for iS, spikes in enumerate(self.spikes):
                if verbose:
                    dsp.update(f'cluster {iS}/{len(self.spikes)}')
                mincont = rpqc.slidingRP(
                    spikes, params={'sampleRate': 30000,
                                    'binSizeCorr': 1/30000}
                                         )[1]

                rp_mincont.append(mincont)
            rp_mincont = np.array(rp_mincont)
            np.save(self.probedir / 'min_cont.npy', rp_mincont)

        self.mincont = rp_mincont

        if doPlot:
            hist_mincont = rp_mincont[rp_mincont != np.nan]
            num_nans = len(rp_mincont[rp_mincont == np.nan])
            f = plt.figure(figsize=(3, 3))
            plt.hist(hist_mincont, bins=20, rwidth=0.9)
            # plt.title(f'excluded {num_nans} nans')
            plt.xlabel('minimum contamination')

        rp_pass = rp_mincont < threshold
        print(f'{np.sum(rp_pass)}/{len(self.neurons)} neurons passed')
        self.rp_pass = rp_pass

        self.neurons = self.neurons[rp_pass]
        self.spikes = self.spikes[rp_pass]
        self.spk_mat = self.spk_mat[self.neurons]

        try:
            self.spk_rate = self.spk_rate[self.neurons]
            self.positions = self.positions[self.neurons]
        except:
            pass

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

    def bin_spikes(self, bins, binsize_s, spks=None, set_self_matrix=False):
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
        # first look for a prebinned version (e.g. if this has been run before)
        binsize_ms = int(binsize_s*1000)
        binned_file = self.probedir / f'binned_{binsize_ms}ms.npy'

        # keep this part even if loading from file -
        #   it's fast and useful for rasters
        if spks is None:
            self.separate_spikes()
            spks = self.spikes

        # try to load file first
        try:
            spk_mat = np.load(binned_file)
            if set_self_matrix:
                self.spk_mat = spk_mat
                return None
            else:
                return spk_mat

        except:
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

            np.save(binned_file, spk_mat)
            if set_self_matrix:
                self.spk_mat = spk_mat
                return None
            else:
                return spk_mat

    def norm_spike_mat(self, cfactor=0.5):
        """
        normalize spiking rates by dividing each cell by its mean plus
        a small factor
        """
        spk_mat = self.spk_mat
        mean_factor = np.mean(spk_mat, axis=1)+cfactor
        spk_mat_norm = spk_mat/mean_factor[:, None]

        self.spk_mat_norm = spk_mat_norm

    def get_spike_rate(self):
        """
        yep
        """
        end_time = np.ceil(np.max(self.npys['spike_times'] / 3e4))
        sec_bins = np.arange(0, end_time, 1)
        spk_rate = np.mean(self.bin_spikes(sec_bins, 1, set_self_matrix=False), axis=1)

        if self.rp_pass is not None:
            spk_rate = spk_rate[self.rp_pass]
        self.spk_rate = spk_rate


    def plot_drift_map(self, by_shank=False):
        probedir = self.probedir
        _plot_drift_map(probedir, by_shank=by_shank)

    def find_max_channels(self):
        templates = np.load(self.probedir / 'templates.npy')
        return _find_max_channels(templates)

    def bin_neurons_positions(self, xbins, ybins):
        maxchans = self.find_max_channels()
        chanpos = np.load(self.probedir / 'channel_positions.npy')
        return _bin_neurons_positions(chanpos[maxchans], xbins, ybins)


def _plot_drift_map(probedir, by_shank=False):

    chanpos = np.load(probedir / 'channel_positions.npy')
    xcoords = chanpos[:, 0]
    ycoords = chanpos[:, 1]

    pc_feat = np.load(probedir / 'pc_features.npy')
    pc_feat = pc_feat[:, 0] # keep first pc only
    pc_feat = np.clip(pc_feat, a_min=1e-6, a_max=None) # close to but not 0... for dividing
    pc_feat_ind = np.load(probedir / 'pc_feature_ind.npy')
    spike_temps = np.load(probedir / 'spike_templates.npy')

    temps = np.load(probedir / 'templates.npy')
    amps = np.load(probedir / 'amplitudes.npy')
    spike_times = np.load(probedir / 'spike_times.npy')

    spike_feat_ind = np.squeeze(pc_feat_ind[spike_temps])
    spike_feat_ycoords = ycoords[spike_feat_ind]
    spike_depths = np.sum(spike_feat_ycoords*np.square(pc_feat), axis=1) // np.sum(np.square(pc_feat), axis=1)
    spike_depths = spike_depths.reshape(-1, 1)

    n_color_bins = 20
    amp_range = np.quantile(amps, [0.1, 0.9])
    amp_bins = np.linspace(amp_range[0], amp_range[1], n_color_bins)
    color_bins = np.linspace(0, 1, n_color_bins)
    colors = mpl.cm.get_cmap('gray_r')

    max_chans = _find_max_channels(temps)

    spike_xcoords = xcoords[max_chans[spike_temps]]
    if by_shank == True:
        x_uni = np.unique(xcoords)
        f = plt.figure(figsize=(8, 4))
        gs = mpl.gridspec.GridSpec(1, 4)

        x_ugroup = x_uni.reshape(4, 2)

        for i, cols in enumerate(x_ugroup):
            spks_col0 = np.r_[spike_xcoords.ravel() == cols[0]].ravel()
            spks_col1 = np.r_[spike_xcoords.ravel() == cols[1]].ravel()

            pass_spks = spks_col0 | spks_col1
            ax = plt.subplot(gs[i])

            for ib in range(n_color_bins-1):
                plot_spks = np.r_[amps > amp_bins[ib]].ravel() & \
                    np.r_[amps <= amp_bins[ib+1]].ravel() & \
                    pass_spks

                plt.scatter(spike_times[plot_spks]/30000, spike_depths[plot_spks], lw=0, s=3,
                            c=np.r_[colors(color_bins[ib])].reshape(1, -1))

            plt.ylim(np.min(ycoords), np.max(ycoords))
            if i == 0:
                plt.ylabel('depth ($\mu$m)')
                plt.figtext(0.5, 0, 'time (s)')
            else:
                plt.yticks([])
                ax.spines['left'].set_visible(False)
            # plt.ylim(400, 1400)


    else:
        f = plt.figure(figsize=(5, 3))
        for i in range(n_color_bins-1):
            # plot_spks = amps[(amps > color_bins[i]) & (amps <= color_bins[i+1])]
            plot_spks = np.r_[amps > amp_bins[i]].ravel() & np.r_[amps <= amp_bins[i+1]].ravel()
            plt.scatter(spike_times[plot_spks]/30000, spike_depths[plot_spks], lw=0, s=3,
                        c=np.r_[colors(color_bins[i])].reshape(1, -1))

        plt.xlabel('time (s)')
        plt.ylabel('depth ($\mu$m)')

def vis_resp_ttest(spkmat, stim_ixs, binsize_s, prestim_s, poststim_s):
    """
    count up spikes before and after stim times
    run quick paired t-test
    """
    prestim_ix = int(np.round(prestim_s/binsize_s))
    poststim_ix = int(np.round(poststim_s/binsize_s))

    def prepostcounts(a):
        pres = []
        posts = []
        for i in stim_ixs:
            pre_count = np.sum(a[i+prestim_ix:i])
            post_count = np.sum(a[i:i+poststim_ix])

            pres.append(pre_count)
            posts.append(post_count)
        return [pres, posts]

    rmap = map(prepostcounts, spkmat)
    counts = list(rmap)
    all_counts = np.array(counts)

    stat, pvals = scipy.stats.ttest_rel(all_counts[:, 0],
        all_counts[:, 1], axis=1)

    return pvals


def _find_max_channels(templates):
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


def _bin_neurons_positions(max_chanpos, xbins, ybins):
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
    # for future i think histogram is better for this but o well
    x_ix = np.digitize(max_chanpos[:, 0], xbins)
    y_ix = np.digitize(max_chanpos[:, 1], ybins)

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

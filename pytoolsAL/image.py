#!/usr/bin/env python

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pytoolsAL as ptAL
import scipy
from skimage import transform
import tifffile

def percentiles(im, lower=0.5, upper=99.5):
    """

    Args:
        im:
        lower:
        upper:

    Returns:

    """
    return [np.percentile(im, lower), np.percentile(im, upper)]


def make_positive(im):
    """
    Scale image stack to start at 0
    Args:
        im (np.ndarray): input image
    Returns:

        Image with min==0 and no negative values

    """
    if np.min(im) < 0:
        im += abs(np.min(im))

    return im


def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def reconstruct_from_svd(svd_temp, svd_spat, t=None, x=None, y=None, comps=200):
    """
    Reconstruct raw image from svd analysis
    Args:
        svd_temp: temporal components
        svd_spat: spatial components
        t: frame range
        x: x pixel range
        y: y pixel range
        comps: number of svd components to use for reconstr

    Returns:
        recon_im: reconstructed image
    """
    # for now - reconstructing full image in x/y, add sub later

    # if no time range, use all
    if t is None:
        t = np.arange(svd_temp.shape[0])
    # get image shapes
    px, py, _spat_comps = svd_spat.shape

    # setup the reshape
    spat_collapse = (px * py, comps)
    svd_spat_collapse = svd_spat[:, :, :comps].reshape(spat_collapse)

    # multiply out
    recon_im = np.dot(svd_spat_collapse, svd_temp[t, :comps].T)

    # then reshape into pixel space
    recon_im = recon_im.reshape(px, py, len(t))

    # then transpose such that the T dimension is first
    recon_im = recon_im.transpose(2, 0, 1)

    return recon_im


def bandpass_and_hilbert(im, fs, band=[3, 6], verbose=0):
    """
    compute phasemap for image stack
    needs ~1-2 seconds to be accurate

    Args:
        im: array image stack in shape (time, x, y)
        fs: sampling rate

    Returns:
        im_phasemap: image stack with phasemap transform
    """
    # right now - hardcode to 512x512 image
    n_frames = im.shape[0]
    im_r = im.reshape(n_frames, 512*512)

    # butterworth bandpass filter
    if verbose > 0:
        print('applying butterworth filter')
    b, a = scipy.signal.butter(N=2, Wn=np.r_[band]/(fs/2), btype='bandpass', output='ba')
    im_filtered = scipy.signal.lfilter(b, a, im_r, axis=0)

    if verbose > 0:
        print('performing hilbert transform')
    # hilbert transform
    im_transform = scipy.signal.hilbert(im_filtered, axis=0)

    if verbose > 0:
        print('converting to phasemap')
    # angle of complex arg in degrees
    im_phasemap = np.angle(im_transform, deg=True)

    # return to 2d
    im_phasemap = im_phasemap.reshape(n_frames, 512, 512)

    return im_phasemap


class FacemapLoader:
    def __init__(self, datapath):
        self.datapath = Path(datapath)

        self.svd_spat = None
        self.svd_temp = None
        self.avg_frame = None

        self.motion = None
        self.roi_x = None
        self.roi_y = None


    def load_facemap(self, doPlot=True, nComps=None):
        datapath = self.datapath

        face_proc = np.load(datapath/'face_proc.npy', allow_pickle=True).item()

        self.svd_spat = face_proc['motMask_reshape'][1]
        self.svd_temp = face_proc['motSVD'][1]
        self.avg_frame = face_proc['avgframe_reshape']

        self.motion = face_proc['motion'][1]
        self.roi_x = face_proc['rois'][0]['xrange_bin']
        self.roi_y = face_proc['rois'][0]['yrange_bin']

        if doPlot:
            self._plot_facemap(nComps)

    def _plot_facemap(self, nComps):
        roi_x = self.roi_x
        roi_y = self.roi_y
        avg_frame = self.avg_frame
        motion = self.motion
        svd_spat = self.svd_spat
        svd_temp = self.svd_temp

        if nComps is None:
            nComps = 3

        f = plt.figure(figsize=(5, (nComps+1)*1.5))
        gs = mpl.gridspec.GridSpec(nComps+1, 3)

        t_lims = (0000, 3000)
        ax = plt.subplot(gs[0, 0])
        plt.imshow(avg_frame[roi_y][:, roi_x], cmap='Greys_r')
        ax = ptAL.plotting.apply_image_defaults(ax)
        plt.title(f'avg frame')

        ax = plt.subplot(gs[0, 1:])
        plt.plot(motion, lw=0.5)
        plt.xlim(t_lims)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x/60:.0f}'))
        plt.xlabel('time (s)')
        plt.title('motion energy')

        for iC in range(nComps):
            ax = plt.subplot(gs[iC+1, 0])
            plt.imshow(svd_spat[..., iC], cmap='plasma')
            plt.title(f'spat comp {iC}')
            ax = ptAL.plotting.apply_image_defaults(ax)

            ax = plt.subplot(gs[iC+1, 1:])
            plt.plot(svd_temp[..., iC], lw=0.5, c='deeppink')
            plt.xlim(t_lims)

            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x/60:.0f}'))
            plt.xlabel('time (s)')
            plt.title(f'temp comp {iC}')

        f.tight_layout()


def scanimage_deinterleave(infile, save=False, outfile=None):
    """
    ScanImage dual-channel .tif files come interleaved - this splits into two channels
    Args:
        infile (str): path to interleaved image
        save (bool): whether or not to save the first (green) channel of the image
        outfile (str): path to save image to

    Returns:

    """
    im = tifffile.imread(infile)

    c1 = im[::2]
    c2 = im[1::2]

    if save:
        tifffile.imsave(outfile, c1, bigtiff=True)
        print('Deinterleaved; saved ch1 to {}'.format(outfile))

    return c1, c2


def scanimage_batch_resave(infile, outfile, n_fr_chunk=300, rewrite_ok=False, downscale_tuple=None):
    """

    Args:
        infile (str): path to input tif
        outfile (str): desired path to output tif
        n_fr_chunk (int): number of frames to load into memory at once
        rewrite_ok (bool): whether or not to rewrite existing tif
        downscale_tuple (tuple, optional): tuple to downscale image by in (z, x, y)

    Returns:
        batched image

    """

    if os.path.isfile(outfile):
        assert rewrite_ok, 'outfile already exists but rewriteOk is False'
        os.remove(outfile)
        print('original outfile exists - deleting.')

    with tifffile.TiffFile(infile) as tif:
        T = len(tif.pages)
        nR, nC = tif.pages[0].shape

    if downscale_tuple is not None:
        nR = int(nR / downscale_tuple[1])
        nC = int(nC / downscale_tuple[2])

    im = np.zeros((T, nR, nC))

    for fr in np.r_[0:T:n_fr_chunk]:
        if fr+n_fr_chunk <= T:
            ix = np.r_[fr:fr + n_fr_chunk]
            chunk = tifffile.imread(infile, key=ix)
            print(fr + n_fr_chunk, end=' ')

            if downscale_tuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscale_tuple)

            im[ix, :, :] = chunk.astype('int16')

        if fr+n_fr_chunk > T:
            ix = np.r_[fr:T]
            chunk = tifffile.imread(infile, key=ix)
            print(T)

            if downscale_tuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscale_tuple)

            im[ix, :, :] = chunk.astype('int16')

    im = im.astype('int16')
    tifffile.imsave(outfile, im, bigtiff=True)
    print('done. saved to {}'.format(outfile))

    return im


def avi_to_gif(path_to_im, fps=30.0):
    """
    Convert .avi movie into .gif (mostly for presentations)
    Later: accept other movie types

    Args:
        path_to_im: existing avi file
        fps: output frame rate of .gif

    Returns:
        None. Saves .avi to .gif with same name.

    """

    impath = path_to_im
    gif_out = impath.replace('.avi', '.gif')

    avi_reader = imageio.get_reader(impath)
    gif_writer = imageio.get_writer(gif_out, fps=fps)

    for im in avi_reader:
        gif_writer.append_data(im)

    gif_writer.close()

    print('Done. Saved gif to {}'.format(gif_out))


def avi_to_tif(path_to_im):
    """
    Convert .avi movie into .tif
    Later: accept other movie types

    Args:
        path_to_im: existing avi file

    Returns:
        None. Saves .avi to .tif with same name.

    """

    impath = path_to_im
    tif_out = impath.replace('.avi', '.tif')

    avi_reader = imageio.get_reader(impath)
    tif_writer = imageio.get_writer(tif_out)

    for im in avi_reader:
        tif_writer.append_data(im)

    tif_writer.close()
    # avi = imageio.mimread(impath)
    # imageio.mimsave(tiff_out, avi)

    print('Done. Saved tif to {}'.format(tif_out))


def get_response_map_1stim(im, n_reps, base_frs, stim_frs):
    """
    Create df/f map from image stack with one repeated stim
    Returns:

    """
    print('stack shape: {}'.format(im.shape))
    nR, nC = im.shape[-2:]

    assert im.shape[0] % n_reps == 0, 'Image length must be divisible by nReps'
    n_frs_per_rep = int(im.shape[0] / n_reps)

    im_r = im.reshape(n_reps, n_frs_per_rep, nR, nC)
    im_avg = np.mean(im_r, axis=0)

    im_avg = make_positive(im_avg)

    f0 = np.mean(im_avg[base_frs, :], axis=0)
    stim_fluo = np.mean(im_avg[stim_frs, :], axis=0)

    dff_2d = 100*(stim_fluo-f0)/f0
    dff_mov = 100*(im_avg-f0)/f0

    return dff_2d, f0, dff_mov

#!/usr/bin/env python

import os

import imageio
import numpy as np
import tifffile
from skimage import transform


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

    avi = imageio.mimread(impath)
    imageio.mimsave(gif_out, avi, fps=fps)

    print('Done. Saved gif to {}'.format(gif_out))


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

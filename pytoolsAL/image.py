#!/usr/bin/env python

import os

import imageio
import numpy as np
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import transform

a = np.asarray
r_ = np.r_


def makePositive(im):
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


def SI_deinterleave(infile, save=False, outfile=None):
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


def SI_batch_resave(infile, outfile, nFrChunk=300, rewriteOk=False, downscaleTuple=None):
    """

    Args:
        infile (str): path to input tif
        outfile (str): desired path to output tif
        nFrChunk (int): number of frames to load into memory at once
        rewriteOk (bool): whether or not to rewrite existing tif
        downscaleTuple (tuple, optional): tuple to downscale image by in (z, x, y)

    Returns:
        batched image

    """

    if os.path.isfile(outfile):
        assert rewriteOk, 'outfile already exists but rewriteOk is False'
        os.remove(outfile)
        print('original outfile exists - deleting.')

    with tifffile.TiffFile(infile) as tif:
        T = len(tif.pages)
        nR,nC = tif.pages[0].shape

    if downscaleTuple is not None:
        nR = int(nR/downscaleTuple[1])
        nC = int(nC/downscaleTuple[2])

    im = np.zeros((T,nR,nC))

    for fr in r_[0:T:nFrChunk]:
        if fr+nFrChunk <= T:
            ix = r_[fr:fr+nFrChunk]
            chunk = tifffile.imread(infile,key=ix)
            print(fr+nFrChunk, end=' ')

            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)

            im[ix,:,:] = chunk.astype('int16')

        if fr+nFrChunk > T:
            ix = r_[fr:T]
            chunk = tifffile.imread(infile, key=ix)
            print(T)

            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)

            im[ix,:,:] = chunk.astype('int16')

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

    imPath = path_to_im
    gifOut = imPath.replace('.avi', '.gif')


    avi = imageio.mimread(imPath)
    imageio.mimsave(gifOut, avi, fps=fps)

    print('Done. Saved gif to {}'.format(gifOut))


def get_response_map_1stim(im, nReps, baseFrs, stimFrs):
    """
    Create df/f map from image stack with one repeated stim
    Returns:

    """
    print('stack shape: {}'.format(im.shape))
    nR, nC = im.shape[-2:]

    assert im.shape[0] % nReps == 0, 'Image length must be divisible by nReps'
    nFrsPerRep = int(im.shape[0]/nReps)

    imR = im.reshape(nReps, nFrsPerRep, nR, nC)
    imAvg = np.mean(imR, axis=0)

    imAvg = makePositive(imAvg)

    f0 = np.mean(imAvg[baseFrs, :], axis=0)
    stimF = np.mean(imAvg[stimFrs, :], axis=0)

    dff_2d = 100*(stimF-f0)/f0
    dff_mov = 100*(imAvg-f0)/f0

    return dff_2d, f0, dff_mov

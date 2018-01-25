# TODO : populate after development is done
from nilearn import plotting as nplt
from .utils import unmask_img


def plot_anat(anat_img, **kwargs):
    nplt.plot_anat(anat_img, **kwargs)
    return


def plot_epi(epi_img, **kwargs):
    nplt.plot_epi(epi_img, **kwargs)
    return


def plot_stat_map(stat_img, bg_img, **kwargs):
    nplt.plot_stat_map(stat_img, bg_img, **kwargs)
    return


def plot_roi(roi, bg, **kwargs):
    nplt.plot_roi(roi, bg, **kwargs)
    return


def plot_connectome(mat, coords, **kwargs):
    nplt.plot_connectome(mat, coords, **kwargs)
    return


def plot_masked(mat, mask, **kwargs):
    nplt.plot_stat_map(unmask_img(mat, mask), **kwargs)
    return

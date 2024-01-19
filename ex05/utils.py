'''
A package containing utility functions for computational MRI exercises.

Author          : Jinho Kim
Email           : jinho.kim@fau.de
First created   : Nov. 2021
Last update     : Nov. 07. 2023
'''
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.patches import Rectangle
from numpy.fft import fftshift, ifftshift, ifft2, fft2
from pathlib import Path
from typing import Optional, Tuple
from scipy.linalg import fractional_matrix_power, pinv
from skimage.metrics import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity as SSIM,
)

def _get_root():
    return Path(__file__).parent

def load_data(fname):
    root = _get_root()
    assert fname.endswith(".mat"), "The filename must contain the .mat extension"
    mat = scipy.io.loadmat(root / fname)
    return mat

def imshow(
    imgs: list,
    gt: Optional[np.ndarray] = None,
    titles: Optional[list] = None,
    suptitle: Optional[str] = None,    
    root: Optional[Path] = None,
    filename: Optional[str] = None,
    fig_size: Optional[tuple] = None,    
    save_indiv: bool = False,    
    num_rows: int = 1,
    pos: Optional[list] = None,
    norm: float = 1.0,
    is_mag: bool = True,
    font_size: int = 15,
    font_color="yellow",
    font_weight="normal",  
    watermark: bool = False,
):
    """
    This function displays multiple images in a single row.
    Args:
        imgs:                       list of images to display
        gt:                         ground truth image, optional
        titles:                     list of titles for each image, optional
        suptitle:                   main title for the figure, optional        
        root:                       Root path to save, optional
        filename:                   name of the file to save the figure, optional
        fig_size:                   figure size, default is (15,10)        
        save_indiv:                 Save individual images or not        
        num_rows:                   The number of rows of layout (a single row by default)
        pos:                        Position of images.
                                    ex) for 2x3 layout, [1,1,1,0,1,1] plots images like
                                                        ====================
                                                        img1    img2    img3
                                                                img4    img5
                                                        ====================
                                    ex) for 2x3 layout with gt given, [1,1,1,0,1,1] plots images like
                                                        ========================
                                                        gt  img1    img2    img3
                                                                    img4    img5
                                                        ========================
        norm:                       normalization factor, default is 1.0
        is_mag:                     plot images in magnitude scale or not (optional, default=True)
        font_size:                  font size for metric display, default is 20
        font_color:                 font color for metric display, default is yellow
                                    Available options are ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        font_weight:                font weight for metric display, default is normal.
                                    Available options are ['normal', 'bold', 'heavy', 'light', 'ultralight', 'medium', 'semibold', 'demibold']
        watermark:                  Add watermark or not
    """

    pos, num_cols = _get_pos(pos, num_rows=num_rows, num_imgs=len(imgs))

    # if gt is given, add 0 to the pos list
    if gt is not None:
        pos_tmp = []
        for i in range(num_rows):
            pos_tmp += [1 if i == 0 else 0] + pos[
                (len(pos) // num_rows) * i : (len(pos) // num_rows) * (i + 1)
            ]
        num_cols = num_cols + 1
        pos = pos_tmp.copy()

    if fig_size is None:
        fig_size = (num_cols * 5, num_rows * 4 + 0.5)

    f = plt.figure(figsize=fig_size)
    titles = [None] * len(imgs) if titles is None else titles
    titles = ["Ground truth"] + titles if gt is not None else titles

    imgs = [gt] + imgs if gt is not None else imgs
    imgs = [np.abs(i) for i in imgs] if is_mag else imgs

    img_idx = 0
    for i, pos_indiv in enumerate(pos, start=1):
        ax = f.add_subplot(num_rows, num_cols, i)

        if pos_indiv == 0:
            img = np.ones_like(imgs[0], dtype=float)
            title = ""
        else:
            img = imgs[img_idx]
            title = titles[img_idx]
            img_idx += 1
        
        if gt is not None and i == 1:
            annotate_gt(ax, font_size, font_color, font_weight)

        if gt is not None and i > 1 and pos_indiv:            
            annotate_metrics(imgs[0], img, ax, font_size, font_color, font_weight)

        norm_method = clr.Normalize() if norm == 1.0 else clr.PowerNorm(gamma=norm)    
        ax.imshow(img, cmap="gray", norm=norm_method)
        ax.axis("off")
        ax.set_title(title)

        if pos_indiv and watermark:
            add_watermark(ax)

    f.suptitle(suptitle) if suptitle is not None else f.suptitle("")

    if root is None:
        root = _get_root()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Results"
    if not root.exists() and filename:
        root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        plt.show()

    elif filename is not None:
        filename = Path(filename).stem
        print(f"Saving figure to {root}")
        plt.savefig(root / filename, bbox_inches="tight", pad_inches=0.3)
        plt.close(f)
        if save_indiv:
            for img, title in zip(imgs, titles):
                img = abs(img)
                title = title.split("\n")[0]
                plt.imshow(img, cmap="gray", norm=clr.PowerNorm(gamma=norm))
                plt.axis("off")
                plt.savefig(
                    root / f"{filename}_{title}",
                    bbox_inches="tight",
                    pad_inches=0.2,
                )


def _get_pos(pos, num_rows, num_imgs):
    num_cols = np.ceil(num_imgs / num_rows).astype(int)
    len_pos = num_rows * num_cols

    if pos is None:
        pos = [1] * num_imgs + [0] * (len_pos - num_imgs)
    else:  # if pos is given
        assert np.count_nonzero(pos) == num_imgs, "Givin pos are not matched to the number of given images"
        res = len_pos - len(pos)
        pos += [0] * res

    return pos, num_cols


def fft2c(x, axes=(-2, -1)):
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)

def normalization(img):
    img = abs(img)
    img -= img.min()
    return img / img.max()

def calc_psnr(x, y):
    x = normalization(x)
    y = normalization(y)
    return PSNR(x, y, data_range=x.max() - x.min())


def calc_ssim(x, y):
    x = normalization(x)
    y = normalization(y)
    return SSIM(x, y, data_range=x.max() - x.min())
    

def annotate_gt(ax, font_size, font_color, font_weight):
    text = f"PSNR[dB]\nSSIM [%]"
    font_props = {'family': 'monospace'} 

    ax.annotate(
        text,
        xy=(1, 1),
        xytext=(-2, -2),
        fontsize=font_size,
        color=font_color,
        xycoords="axes fraction",
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontweight=font_weight,
        fontproperties=font_props
    )


def annotate_metrics(trg, src, ax, font_size, font_color, font_weight):
    psnr = calc_psnr(trg, src)
    ssim = calc_ssim(trg, src)

    text = f"{psnr:.2f}\n{ssim * 100:.2f}"

    ax.annotate(
        text,
        xy=(1, 1),
        xytext=(-2, -2),
        fontsize=font_size,
        color=font_color,
        xycoords="axes fraction",
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontweight=font_weight,
    )


def add_watermark(ax):    
    ax.annotate(
        "CMRI 23WS",
        xy=np.random.uniform(0.2, 0.8, 2),        
        fontsize=30,
        color='white',
        alpha=0.5,
        xycoords="axes fraction",
        fontweight="normal",
    )

def cmplx_sum(m):
    return np.sum(m, axis=-1)

def apply_psi(x, psi):
    '''
    This function applies the coil noise covariance matrix to the image
    \Psi^{-1/2}x

    @parak
    x:          matrix of shape [nPE,nFE,nCh]
    psi:        matrix of shape [nCh,nCh]

    @return:
    y:          matrix of shape [nPE,nFE,nCh]
    '''
    # assert if the dimensions are correct
    assert x.shape[-1] == psi.shape[0], "The last dimension of x must be equal to the first dimension of psi"    
    psi = fractional_matrix_power(psi, -1 / 2)

    return (psi @ x.transpose(0, 2, 1)).transpose(0, 2, 1)


def get_alias_idx(PE, R, locs):
    '''
    Get an index for aliased image among indices in locs
    @param PE: Length of phase encoding
    @param R: Acceleration factor
    @param locs: indices for SENSE reconstruction at one point
    @return: an index for aliased image
    '''
    alias_PE = np.arange(0, PE, R).size
    min_idx = PE // 2 - alias_PE // 2
    max_idx = PE // 2 + alias_PE // 2
    for loc in locs:
        if min_idx <= loc < max_idx:
            return loc - min_idx
    else:
        assert idx_PE_alias is not None


    
def calc_g(C):
    """
    Calculate g-factor
    @param C:           \Psi^{-1/2} * sens_maps
    @return:            g-factor
    """
    g = np.sqrt(np.real(np.diag(pinv(np.conj(C.T) @ C)) * np.diag(np.conj(C.T) @ C)))
    return g
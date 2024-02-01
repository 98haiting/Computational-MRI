from typing import Optional
from datetime import datetime
from pathlib import Path
from skimage.metrics import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity as SSIM,
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pickle


def _get_root():
    return Path(__file__).parent

def plot(
        vectors: list,
        labels: Optional[list] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        smoothing: Optional[int] = 1,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        root: Optional[Path] = None,
        filename: Optional[str] = None,
        color: str = "white",
        watermark: Optional[bool] = False,
    ):
    """
    This function plots multiple vectors in a single figure.
    Args:
        vectors:            list of images to display
        labels:             list of labels for each image, optional
        xlabel:             label for x-axis, optional
        ylabel:             label for y-axis, optional
        title:              title for the figure, optional
        smoothing:          smoothing factor for the plot, optional
        xlim:               x-axis limits, optional
        ylim:               y-axis limits, optional
        root:               Root path to save, optional
        filename:           name of the file to save the figure, optional        
        color:              color of the watermark, default is white
        watermark:          Add watermark or not
    """
    if not isinstance(vectors, list):
        vectors = [vectors]

    f, a = plt.subplots(1)
    for vector, label in zip(vectors, labels):
        a.plot(np.convolve(vector, np.ones((smoothing,)) / smoothing, mode='valid'), label=label)

    if xlabel:
        a.set_xlabel(xlabel)
    if ylabel:
        a.set_ylabel(ylabel)
    if title:
        a.set_title(title)
    if xlim:
        a.set_xlim(xlim)
    if ylim:
        a.set_ylim(ylim)
    if watermark:
        add_watermark(a, color=color)
    a.legend()

    if root is None:
        root = _get_root()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Results"
    if not root.exists() and filename:
        root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        plt.show()
    else:
        plt.savefig(root/f"{filename}_{title}", bbox_inches="tight", pad_inches=0.2)
    plt.close()

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
    imgs = anything_to_np(imgs)
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
        plt.savefig(root / filename, bbox_inches="tight", pad_inches=0.1)
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
                    pad_inches=0.1,
                )

def anything_to_np(imgs):
    is_np = False
    if not isinstance(imgs, list):
        imgs = [imgs]
        is_np = True

    for i in range(len(imgs)):
        img = imgs[i].squeeze()

        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()

        if img.shape[-1] == 2:  # real-valued img (..., 2)
            img = img[..., 0] + 1j * img[..., -1]

        img = np.flip(img)
        imgs[i] = np.abs(img)

    if is_np:
        imgs = imgs[0]
    return imgs


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
    
def add_watermark(ax, color='white'):    
    ax.annotate(
        "CMRI 23WS",
        xy=np.random.uniform(0.2, 0.8, 2),        
        fontsize=30,
        color=color,
        alpha=0.5,
        xycoords="axes fraction",
        fontweight="normal",
    )

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
    
def pprint(text: str, level: int = 0):
    cur_time = datetime.now()
    text = " " * (level * 2) + text
    print(f"[{cur_time.strftime('%H:%M:%S')}] {text}")
    

def normalize(x: np.ndarray):
    """
    Normalizes the given data to zero mean and unit standard deviation.

    Args:
        x: data
    Returns:
        normalized data
    """
    return (x - np.mean(x)) / np.std(x)


def get_save_path(folder_name: Optional[str]):
    """
    Get a save path for a given directory. The save path is the current date and time.

    Returns:
        save path
    """
    data_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = folder_name if folder_name is not None else data_str

    save_path = Path.cwd() / "logs" / folder_name

    if not save_path.exists():
        save_path.mkdir(parents=True)
    return save_path


def load_loss(save_dir, model_name):
    with open(save_dir/f'{model_name}_train_loss_pretrained.pkl', 'rb') as trn_loss, open(save_dir/f'{model_name}_validation_loss_pretrained.pkl', 'rb') as val_loss:
        train_loss = pickle.load(trn_loss)
        validation_loss = pickle.load(val_loss)

    return train_loss, validation_loss
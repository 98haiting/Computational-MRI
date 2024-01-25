'''
A package containing utility functions for computational MRI exercises.

Author          : Jinho Kim
Email           : jinho.kim@fau.de
First created   : Nov. 2021
Last update     : Nov. 22. 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from pathlib import Path
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder

def _get_root():
    return Path(__file__).parent

def load_data_ex1(path: Union[Path, str] = None):
    data = []
    if path is None:
        path = _get_root() / "data" / "dti"
    
    for fname in (path).iterdir():
        if fname.suffix == ".csv":
            data.append(pd.read_csv(fname, header=None).values)
    
    return data

def load_data_ex2(path: Union[Path, str] = None):
    entire_data = []
    if path is None:
        path = _get_root() / "data" / "recon_classification"
    
    for data_split in ["train", "val"]:  
        images, labels = [], []
        for root, dir, files in os.walk(path/data_split):
            root = Path(root)
            for file in files:
                if file.endswith('.png'):                
                    image = cv2.imread(str(root/file))
                    image = rgb2gray(image)
                    [nR, nC] = image.shape
                    image = image.reshape(1, nR, nC)
                    label = root.stem
                    images.append(image)
                    labels.append(label)
        entire_data.append(images)
        entire_data.append(labels)

    x_train, y_train, x_val, y_val = entire_data

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.fit_transform(y_val)

    return np.array(x_train), y_train, np.array(x_val), y_val

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

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
        add_watermark(a, color="black")
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

def add_watermark(ax, color="white"):    
    ax.annotate(
        "CMRI 23WS",
        xy=np.random.uniform(0.2, 0.8, 2),        
        fontsize=30,
        color=color,
        alpha=0.5,
        xycoords="axes fraction",
        fontweight="normal",
    )
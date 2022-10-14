import matplotlib.pyplot as plt
import numpy as np

import time
import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

import pdb

class CNN_Helper():
    def __init__(self, **params):
        return

    def create_img(self):
        h, w = 8, 8
        img = np.zeros((h,w))
        img[2:-2,1] = 1
        img[-2,2:-2]= 1

        return img

        
    def create_filters(self):
        filt_horiz  = np.zeros( (3,3) )
        filt_horiz[0,:] = 1

        filt_vert = np.zeros( (3,3) )
        filt_vert[:,0] = 1

        filt_edge_h  = np.zeros( (3,3) )
        filt_edge_h[0,:] = -1
        filt_edge_h[1,:] =  1
        filt_edge_h[2,:] =  -1

        filt_edge_v  = filt_edge_h.T

        filt_edge_2 = np.zeros( (3,3))
        filt_edge_2[:, 0]  = 1
        filt_edge_2[-1, :] = 1

        filters = { "horiz, light to dark": filt_horiz,
                    "vert,  light to dark": filt_vert,
                    "horiz, light band":    filt_edge_h,
                    "vert, light band":     filt_edge_v,
                    "L"               :     filt_edge_2
                    }
            
        return filters

    def showmat(self,mat, ax, select=None, show_all=False):
        ax.matshow(mat, cmap="gray")

        if show_all:
            for row in range(0, mat.shape[0]):
                for col in range(0, mat.shape[1]):
                    ax.text(col, row, mat[row, col], color='black', backgroundcolor="white", ha='center', va='center')

        if select is not None:
            row_min, row_max, col_min, col_max = select
            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    ax.text(col, row, mat[row, col], color='white', backgroundcolor="blue", ha='center', va='center')



    def pad(self, img, pad_size):
        padded_img = np.zeros( list(s+ 2*pad_size for s in img.shape) )
        padded_img[ pad_size:-pad_size, pad_size:-pad_size] = img
        return padded_img


    # Note: score[row,col] is result of applying filter CENTERED at img[row,col]
    def apply_filt_2d(self, img, filt):
        # Shape of the filter
        filt_rows, filt_cols = filt.shape

        # Pad the image
        pad_size = (filt_rows-1)// 2
        padded_img = self.pad(img, pad_size)

        score = np.zeros( img.shape )
        for row in range( img.shape[0] ):
            for col in range( img.shape[1] ):
                # window, centered on (row,col)of img
                # - is centered at (row+pad_size, col+pad_size) in padded_img
                # - so corners are (row+pad_size - pad_size, col+pad_size - pad_size)
                window = padded_img[ row:row+filt_rows, col:col+filt_cols]
                score[row,col] = (window *filt).sum()

        return score

    def plot_convs(self, img=None, filters=None):
        if filters is None:
            filters= self.create_filters()

        if img is None:
            img = self.create_img()
            
        fig, axs = plt.subplots( len(filters), 3, figsize=(12, 12 ),
                                 gridspec_kw={'width_ratios': [8, 1, 8]}
                                 )

        fig.subplots_adjust(hspace=10)

        i = 0
        for lab, filt in filters.items():

            img_filt = self.apply_filt_2d(img, filt)
            _= axs[i,0].matshow(img, cmap="gray")
            _= axs[i,0].xaxis.set_ticks_position('bottom')
            _= axs[i,0].set_title ("input")
            

            _= axs[i,1].matshow(filt, cmap="gray")
            _= axs[i,1].set_title ("filter:\n" + lab)
            _= axs[i,1].xaxis.set_ticks_position('bottom')
            
            _= axs[i,2].matshow(img_filt, cmap="gray")
            _= axs[i,2].xaxis.set_ticks_position('bottom')
            _= axs[i,2].set_title ("convolution output")
            
            i += 1

        fig.tight_layout()
        
        return fig, axs

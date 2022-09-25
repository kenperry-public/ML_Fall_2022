import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pickle
import math

import os
import time

from sklearn.datasets import fetch_openml, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA



import mnist_helper as mnhelp

class PCA_Helper():
    def __init__(self,  save_dir="/tmp", visible=True, **params):
        self.save_dir = save_dir
        self.visible = visible
        
        return

    def mnist_init(self):
        mnh = mnhelp.MNIST_Helper()
        self.mnh = mnh

        X, y = mnh = mnh.fetch_mnist_784()

        return X, y

    def mnist_PCA(self, X, n_components=0.95, **params):
        """
        Fit PCA to X

        Parameters
        ----------
        n_components: number of components
        - Passed through to sklearn PCA
        -- <1: interpreted as fraction of explained variance desired
        -- >=1 interpreted as number of components
        
        """
        if n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA()

        pca.fit(X)

        return pca

    def transform(self, X,  model):
        """
        Transform samples through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features)
        model: sklearn model object, e.g, PCA

        X_reduced: ndarray (num_samples, pca.num_components_)
        """
        X_transformed = model.transform(X)
        return X_transformed

    def inverse_transform(self, X,  model):
        """
        Invert  samples that were transformed through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features_trasnformed)
        model: sklearn model object, e.g, PCA

        X_reconstruct: ndarray (num_samples, num_features)
        """
        X_reconstruct = model.inverse_transform(X)
        return X_reconstruct

    def num_components_for_cum_variance(self, pca, thresh):
        """
        Return number of components of PCA such that cumulative variance explained exceeds threshhold

        Parameters
        ----------
        pca: PCA object
        thresh: float. Fraction of explained variance threshold
        """

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= thresh) + 1

        return d

    def plot_cum_variance(self, pca):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        x  = range(1, 1 + cumsum.shape[0])
        
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        _ = ax.plot(x, cumsum)

        _ = ax.set_title("Cumulative variance explained")
        _ = ax.set_xlabel("# of components")
        _ = ax.set_ylabel("Fraction total variance")

        _= ax.set_yticks( np.linspace(0,1,11)  )

        return fig, ax

    def mnist_filter(self, X, y, digit):
        cond = (y==digit)
        X_filt = X[ cond ]
        y_filt = y[ cond ]

        return X_filt, y_filt

    def mnist_plot_2D(self, X, y):
        fig, ax = plt.subplots(1,1, figsize=(12,6))

        cmap="jet"
        _ = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
        # _ = ax.axis('off')

        _ = ax.set(xlabel='component 1', ylabel='component 2')

        
        norm = mpl.colors.Normalize(vmin=0,vmax=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # _ = plt.colorbar(sm)

        return fig, ax

    def pca_charts(self):
        save_dir = self.save_dir
        visible = self.visible

        # Get data, split into train/test
        X_mnist, y_mnist = self.mnist_init()
        X_mnist_train, X_mnist_test, y_mnist_train, y_mnist_test = train_test_split(X_mnist, y_mnist)

        # Perform PCA
        pca_mnist = self.mnist_PCA(X_mnist_train)
        X_mnist_train_reduced = self.transform(X_mnist_train, pca_mnist)

        # Plot cumulative variance
        fig_cum, ax = self.plot_cum_variance(pca_mnist)

        cum_var_file = os.path.join(save_dir, "cum_var.png")
        
         # Save plot
        fig_cum.savefig(cum_var_file)

        if not visible:
            plt.close(fig_cum)

        return { "cumulative variance": cum_var_file }
        
    def corr_features_charts(self):
        save_dir = self.save_dir
        visible = self.visible
        
        m = 30
        rng = np.random.RandomState(1)
        x_1 = rng.rand(m)
        x_2 = 2 * x_1

        fig_perf, ax = plt.subplots(1,1,figsize=(12,4))
        _= ax.scatter( x_1, x_2, color="blue", s=50)
        _= ax.plot( x_1, x_2, color="black", linestyle="dashed")
        _= ax.set_xlabel("$x_1$")
        _= ax.set_ylabel("$x_2$")

        perf_corr_file = os.path.join(save_dir, "features_perf_corr.png")
        
        # Save plot
        fig_perf.savefig(perf_corr_file)

        eps = .01
        x_2p = 2 * x_1 + .2 * rng.randn( x_1.shape[0] )

        x_p = np.concatenate( [ x_1.reshape(-1,1), x_2p.reshape(-1,1)], axis=1)

        fig_imperf, ax = plt.subplots(1,1,figsize=(6,6))
        _= ax.scatter( x_p[:,0], x_p[:,1], color="blue", s=50)
        _= ax.plot( x_1, x_2, color="black", linestyle="dashed")
        _= ax.set_xlabel("$x_1$")
        _= ax.set_ylabel("$x_2$")
        _= ax.axis("equal")

        imperf_corr_file = os.path.join(save_dir, "features_imperf_corr.png")

        # Save plot
        fig_imperf.savefig(imperf_corr_file)

        pca_x2p = PCA()
        #x_p = x_p - x_p.mean(axis=0)

        pca_x2p_proj = pca_x2p.fit_transform(x_p)


        def draw_vector(v0, v1, ax=None):
            arrowprops=dict(arrowstyle='->',
                            linewidth=2,
                            color="black",
                            shrinkA=0, shrinkB=0)
            _ = ax.annotate('', v1, v0, arrowprops=arrowprops)

            return ax

        fig_basis, ax = plt.subplots(1,1, figsize=(6,6))
        mean = x_p.mean(axis=0)

        maxp = np.sqrt( pca_x2p.explained_variance_[-1] )

        _= ax.scatter( x_p[:,0], x_p[:,1], color="blue", s=10)

        for i in range(0, 2):
            comp, length = pca_x2p.components_[i], pca_x2p.explained_variance_[i]
            v = comp  # *  np.sqrt(length)   
            _= draw_vector( mean, mean + v , ax=ax)

        _= ax.scatter( mean[0], mean[1], s=50, color="black")

        _= ax.axis("equal")

        basis_file = os.path.join(save_dir, "features_basis.png")

        # Save plot
        fig_basis.savefig(basis_file)

        if not visible:
            plt.close(fig_perf)
            plt.close(fig_imperf)
            plt.close(fig_basis)

            return { "perfect corr": perf_corr_file,
                     "imperfect corr": imperf_corr_file,
                     "basis": basis_file
                     }
            
class VanderPlas():
    def __init__(self, save_dir="/tmp", visible=True, **params):
        self.save_dir = save_dir

        self.visible = visible
        
        return

    """
    The following set of methods illustrate PCA and are drawn from 
    external/PythonDataScienceHandbook/notebooks/05.09-Principal-Component-Analysis.ipynb#Introducing-Principal-Component-Analysis
    """
    def create_data(self):
        rng = np.random.RandomState(1)
        X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

        rng = np.random.RandomState(1)
        X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

        return X


    def draw_vector(self, v0, v1, ax=None):
        arrowprops=dict(arrowstyle='->',
                        linewidth=2,
                        color="black",
                        shrinkA=0, shrinkB=0)
        _ = ax.annotate('', v1, v0, arrowprops=arrowprops)

        return ax

                   
    def show_2D(self, X, whiten=False, alpha=0.4, points=[]):
        """
        Plot the dataset (X) and show the PC's, both in original feature space and transformed (PC) space

        Parameters
        ----------
        X: feature matrix

        whiten: Boolean,  whiten argument to PCA constructor 
        alpha: alpha for scatter plot (plt.scatter argument)
        """

        pca = PCA(n_components=2, whiten=whiten)
        self.pca = pca
        
        pca.fit(X)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        _ = fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        # plot data in original feature space
        _ = ax[0].scatter(X[:, 0], X[:, 1], alpha=alpha)

        # Show the components (i.e, PC's, axes) in original feature space
        # Note: PCA
        # - works on centered X data
        # - automatically performs the centering, before performing SVD
        # - So pca.components_ are the axes of the CENTERED data, i.e, with mean pca.mean_
        # - When transforming a test example x, pca.transform will center it before applying the transformation matrix
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)
            self.draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])

        # Unless the figure height and width are identical
        # - even if horizontal and vertical scales have same limits
        # - the aspect ratio difference will make the orthogonal vectors LOOK non-orthogonal
        # - make the axes distance comparable
        ax[0].axis('equal');
        
        # _ =ax[0].set(xlabel='x', ylabel='y', title='input')
        _= ax[0].set_title("Original")
        _= ax[0].set_xlabel("$x_1$")
        _= ax[0].set_ylabel("$x_2$")
        
        # plot data in transformed (PC) space
        X_pca = pca.transform(X)
        _ = ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=alpha)

        # Show the axes in transformed (i.e, rotated) feature space
        if whiten:
            self.draw_vector([0, 0], [0, 3], ax=ax[1])
            self.draw_vector([0, 0], [3, 0], ax=ax[1])
        else:
            for length, vector in zip(pca.explained_variance_, [np.array([1,0]), np.array([0,1]) ] ):
                v = vector * 3 * np.sqrt(length)
                self.draw_vector([0,0], [0,0] + v, ax=ax[1])

        ax[1].axis('equal')
        _ = ax[1].set(xlabel='component 1', ylabel='component 2',
                      title='principal components',
                      # xlim=(-5, 5), ylim=(-3, 3.1)
                     )
        _= ax[1].set_xlabel("$\\tilde{x}_1$")
        _= ax[1].set_ylabel("$\\tilde{x}_2$")


        # Map a few specific points
        if len(points) > 0:
            points_pca = pca.transform( points )
            _= ax[0].scatter(points[:,0], points[:,1], c="red")
            _= ax[1].scatter(points_pca[:,0], points_pca[:,1], c="red")
            
        fig.tight_layout()

    """
    The following methods show dimensionality reduction using PCA on sklearns "digits" dataset (low resolution digits).
    It is derived from:
    http://localhost:8888/notebooks/NYU/external/PythonDataScienceHandbook/notebooks/05.09-Principal-Component-Analysis.ipynb#PCA-for-visualization:-Hand-written-digits
    """

    def digits_plot(self, data, save_file=None, title=None):
        """
        Plot the data from the digits dataset (each digit is 8x8 matrix of pixels)
        
        Parameters
        ----------
        data: List.  Elements are digits (8x8 matrices)
        """
        fig, axes = plt.subplots(4, 10, figsize=(12, 6),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        for i, ax in enumerate(axes.flat):
            _ = ax.imshow(data[i].reshape(8, 8),
                      cmap='binary', interpolation='nearest',
                      clim=(0, 16))

        if title is not None:
            fig.suptitle(title)
      
        # Save plot
        if save_file is not None:
            fig.savefig(save_file)
            
        return fig, axes

    def digits_reconstruction(self, data, n_components=None, save_file=None, visible=True, title=None):
        """
        Transform data via PCA using num_components PC's, and show reconstruction
        """
        if n_components is not None:
            pca = PCA(n_components)  # project from 64 to 2 dimensions
        else:
            pca = PCA()

        self.pca = pca
        
        projected = pca.fit_transform(data)

        reconstructed = pca.inverse_transform(projected)
        fig_digits, axes_digits = self.digits_plot(reconstructed, save_file, title=title)

        if title is not None:
            fig_digits.suptitle(title)
        
        if not visible:
            plt.close(fig_digits)
            
        return pca, projected, reconstructed

    def digits_show_clustering(self, projected, targets, alpha=0.9, save_file=None, visible=True, cmap_name="plasma", title=None):
        fig, ax = plt.subplots( figsize=(10,6) )

        # cmap=plt.cm.get_cmap('cividis', 10)
        cmap=plt.cm.get_cmap(cmap_name, 10)

        ax.scatter(projected[:, 0], projected[:, 1],
                    c=targets, edgecolor='none', alpha=alpha,
                    cmap=cmap)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')

        norm = mpl.colors.Normalize(vmin=0,vmax=9)

        fig.colorbar( plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax )

        if title is not None:
            fig.suptitle(title)
            
        if save_file is not None:
            fig.savefig(save_file)

        if not visible:
            plt.close(fig)
            
    def plot_cum_variance(self, pca):
        pch = PCA_Helper()
        return pch.plot_cum_variance(pca)

    # Methods to explore subsets of the digits dataset
    def digits_subset(self, digits, subset, n_components=2, save_file=None, visible=True):
        """
        Parameters
        ----------
        digits: sklearn.utils.Bunch.  The digits data bunch
        - digits.data: ndarray of digits data
        - digits.target: labels

        subset: List.  Array of target labels
        """

        # Create subset of digits and targets
        mask = np.isin( digits.target, subset)
        digits_t, targets_t = digits.data[ mask ], digits.target[mask]

        title = "Digits subset: " + str(subset)
        
        # Perform PCA on the subset
        digits_pca_t, digits_projected_t, digits_reconstructed_t = self.digits_reconstruction(digits_t, n_components=n_components, save_file=save_file, visible=visible, title=title)
    
        return digits_projected_t, targets_t

    def digits_subset_components(self, n_show=2):
        fig, axs = plt.subplots(1, n_show, figsize=(12,6))
        data = self.pca.components_[:n_show]

        for i in range(0,n_show):
            ax = axs[i]
            _ = ax.imshow(data[i].reshape(8, 8) > 0,
                                  # cmap='binary', interpolation='nearest',
                                  #clim=(0, 16)
                          cmap="binary"
                         )

        return fig, axs

    def digits_subset_show_clustering(self, projected_t, targets_t, save_file=None, visible=True, title=None):
        m = (targets_t <= 9)
        #m = np.isin( targets_t, [1, 4, 7])
        
        d, t = projected_t[ m ], targets_t[m]
        self.digits_show_clustering(d, t, save_file=save_file, visible=visible, title=title)

    def create_charts(self):
        save_dir = self.save_dir
        visible = self.visible
        
        print("Saving to directory: ", save_dir)
        
        print("Create Digits subset chart")

        digits = load_digits()
        subset1 = [0, 4, 7, 9]

        digits_subset_file = os.path.join(save_dir, "digits_subset.png")
        projected_t, targets_t = self.digits_subset(digits, subset1, n_components=4, save_file=digits_subset_file, visible=visible)

        print("Create Digits subset clustering chart")

        digits_subset_cluster_file = os.path.join(save_dir, "digits_subset_cluster.png")
        print("Number of examples: {n:d}".format(n=projected_t.shape[0]))
        cluster_title = "Digits subset: " + str(subset1)

        self.digits_subset_show_clustering(projected_t, targets_t, save_file=digits_subset_cluster_file, visible=visible, title=cluster_title )

        return { "digits subset": digits_subset_file,
                 "digits subset cluster": digits_subset_cluster_file
                 }

    
class Reconstruct_Helper():
    def __init__(self, **params):
        self.cmap="binary"
        self.num_cols = 4
        return

    def create_data_digits(self, subset=None):
        digits = load_digits()

        data, targets = digits.data, digits.target

        if subset is not None:
            mask = np.isin( digits.target, subset)
            digits_t, targets_t = digits.data[ mask ], digits.target[mask]
    
            data, targets = digits_t, targets_t

        self.data, self.targets = data, targets

    def fit(self, n_components=None):
        data = self.data
        
        if n_components is not None:
            pca =  PCA(n_components=n_components)
        else:
            pca = PCA()

        self.pca = pca

        # Fit the PCA and get the projected (in PC Compoment basis space) features
        dataProj = pca.fit_transform(data)

        self.dataProj = dataProj

    def show_data_comp(self, data_idx=0):
        pca = self.pca
        cmap = self.cmap
        data = self.data
        num_cols = self.num_cols
        
        imshape = (int(pca.components_.shape[-1]**0.5), int(pca.components_.shape[-1]**0.5))
        self.imshape = imshape

        # Show original data
        fig0, ax0 = plt.subplots(1,1, figsize=(12,4))
        _= ax0.imshow( data[data_idx].reshape(imshape), cmap=cmap)

        # Plot the components (proto-typical shapes, since actual shape is weighted combo of components)
        # Show the mean
        figm, axm = plt.subplots(1,1, figsize=(12,4))
        axm.imshow( pca.mean_.reshape(imshape), cmap=cmap)
        _= axm.set_xticks([])
        _= axm.set_yticks([])
        _= axm.set_title("Comp 0 (Mean)")
  

        
        figc, axc = plt.subplots( math.ceil(pca.components_.shape[0]/num_cols), num_cols, figsize=(12,8))

        axc = axc.ravel()
        # Show each component
        for i, comp in enumerate(pca.components_):
            comp = comp # + pca.mean_
            
            _= axc[i].imshow( comp.reshape(imshape), cmap=cmap)
            _= axc[i].set_xticks([])
            _= axc[i].set_yticks([])
            _= axc[i].set_title("Comp {c:d}".format(c=i+1))
    
        return fig0, ax0, figm, axm, figc, axc

    def show_recon(self, data_idx=0, Debug=False):
        pca = self.pca
        cmap = self.cmap
        data = self.data
        num_cols = self.num_cols
        dataProj =self.dataProj
        
        imshape = (int(pca.components_.shape[-1]**0.5), int(pca.components_.shape[-1]**0.5))

        fig, axs = plt.subplots( math.ceil(pca.components_.shape[0]/num_cols), num_cols, figsize=(12,8))

        x_tilde =  dataProj[ data_idx ]
        if Debug:
            print("\\tilde{X}^(data_idx):", x_tilde)
            
        axs = axs.ravel()
        approx = pca.mean_

        # Reconstruct the original, one componenent at a time
        # - approx is the approximation of x at each stage
        for i, comp in enumerate(pca.components_):
            coeff = dataProj[data_idx][i]
            approx = approx + coeff * comp
            _= axs[i].imshow(approx.reshape(imshape), cmap=cmap)
            _= axs[i].set_xticks([])
            _= axs[i].set_yticks([])
            ltx = '\sum_{{j=0}}^{e:d} \\tilde x_j * PC_j'.format(e=i+1)
            _= axs[i].set_title("$" + ltx + "$", pad=20)

            fig.tight_layout()

        # Compare Original and reconstruction
        fig1,ax1 = plt.subplots(1,2, figsize=(12,4))
        _= ax1[0].imshow(data[data_idx].reshape(imshape), cmap=cmap)
        _= ax1[0].set_title("Original")
        _= ax1[1].imshow(approx.reshape(imshape), cmap=cmap )
        _= ax1[1].set_title("Reconstruction")

        # L2 error: original x versus the reconstructed approximation
        error = np.sum( (data[data_idx] - approx)**2 )
        if Debug:
            print("L2 reconstruction error: ", error )

        return fig, axs, fig1, ax1, x_tilde, error


class YieldCurve_PCA():
    """
    Perform PCA on CHANGES in the  yield curve.

    Derived from:
    https://github.com/radmerti/MVA2-PCA/blob/master/YieldCurvePCA.ipynb

    NOTES
    -----
    - The input data seems to be monthly observations of the level of the Yield Curve
    - Another resource does a PCA of the swap yield curve
    https://clinthoward.github.io/portfolio/2017/08/19/Rates-Simulations/

    -- fewer maturities
    -- But gets data from Quandl
    --- Uses FRED so can also get from the via pandas datareader.
    --- Better choice than the unknown csv file above, but would require additional modules

    - Both of the above notebooks do PCA on the LEVEL of the yield curve, NOT the change

    - I have modified it for changes in level.  That is more appropriate to risk management and is in keeping with the
    -- original Litterman Scheinkman paper
    https://www.math.nyu.edu/faculty/avellane/Litterman1991.pdf
    
    """
    def __init__(self, **params):
        YC_PATH = os.path.join("./data", "yield_curve")

        if not os.path.isdir(YC_PATH):
            print("YieldCurve_PCA: you MUST download the data to ", YC_PATH)
            
        self.YC_PATH = YC_PATH

        return

    def create_data(self, csv_file=""):
        if len(csv_file) == 0:
            YC_PATH = self.YC_PATH
            csv_file = os.path.join(YC_PATH, "Marktzinsen_mod.csv")
        
        df = pd.read_csv(csv_file, sep=',')

        df['Datum'] = pd.to_datetime(df['Datum'],infer_datetime_format=True)
        
        df.set_index('Datum', drop=True, inplace=True)
        
        df.index.names = [None]
        
        df.drop('Index', axis=1, inplace=True)
        
        dt = df.transpose()

        return df

    def plot_YC(self, df):
        plt.figure(figsize=(12,6))

        plt.plot(df.index, df)
        plt.xlim(df.index.min(), df.index.max())
        # plt.ylim(0, 0.1)
        plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
        for i in range(df.index.min().year, df.index.max().year+1):
            plt.axvline(x=df.index[df.index.searchsorted(pd.datetime(i,1,1))-1],
                        c="grey", linewidth=0.5, zorder=0)


    def doPCA(self, df, doDiff=True):
        """
        Parameters
        ----------
        doDiff: Boolean.  Take first order difference of data before performing PCA
        """
        # calculate the PCA (Eigenvectors & Eigenvalues of the covariance matrix)
        pcaA = PCA(n_components=3, copy=True, whiten=False)

        # pcaA = KernelPCA(n_components=3,
        #                  kernel='rbf',
        #                  gamma=2.0, # default 1/n_features
        #                  kernel_params=None,
        #                  fit_inverse_transform=False,
        #                  eigen_solver='auto',
        #                  tol=0,
        #                  max_iter=None)

        # transform the dataset onto the first two eigenvectors
        # kjp: change to diff
        df_in = df.copy()

        if doDiff:
            df_in = df_in.diff(axis=0).dropna()


        pcaA.fit(df_in)
        dpca = pd.DataFrame(pcaA.transform(df_in))
        dpca.index = df_in.index


        return pcaA, dpca

    def plot_cum_variance(self, pca):
        pch = PCA_Helper()
        return pch.plot_cum_variance(pca)

    def plot_components(self, pcaA, xlabel="Original feature #" , ylabel="Original feature value"):
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        ax.set_title('First {0} PCA components'.format(np.shape(np.transpose(pcaA.components_))[-1]))

        ax.plot(np.transpose(pcaA.components_) )

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.legend(["PC 1", "PC 2", "PC 3"])

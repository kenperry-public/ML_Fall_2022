import matplotlib.pyplot as plt
import numpy as np

import os
import math

from sklearn import datasets

try:
	from sklearn.datasets.samples_generator import make_circles
except ImportError:
	from sklearn.datasets import make_circles

from sklearn.datasets import make_blobs

from sklearn.svm import SVC
from sklearn import svm
from sklearn import linear_model

from sklearn import pipeline
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.preprocessing import PolynomialFeatures


from matplotlib.colors import ListedColormap

import pdb

import functools

from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

import class_helper

class SVM_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        self.clh = class_helper.Classification_Helper()
        return

    def sigmoid(self, x):
        x = 1/(1+np.exp(-x))
        return x


    def hinge(self, score, target=1, hinge_pt=0):
        """
        Return value of hinge function

        Parameters
        ----------
        score: ndarray.  List of values for which to compute hinge function
        hinge_pt: Integer.  Hinge point
        target: Integer.  +1 for Positive examples, -1 for Negative examples
        """
        return  np.maximum(0,  hinge_pt - np.sign(target) * score)
        
    def plot_pos_examples(self, score, ax, x_axis="Score", hinge_pt=None):
        # Apply sigmoid to turn into probability
        p = self.sigmoid(score)

        neg_logs =  - np.log(p)

        xs, xlabel = score, "Score"
        if x_axis != "Score":
            xs, xlabel = p, x_axis
        
        _= ax.plot(xs, neg_logs, label="- log p")
        _= ax.set_title("Positive examples")
        _= ax.set_xlabel(xlabel)

        if hinge_pt is not None:
            hinge = self.hinge(score, target=1, hinge_pt=hinge_pt)
            _= ax.plot(xs, hinge, label="hinge")
            
        _= ax.legend()


    def plot_neg_examples(self, score, ax,  x_axis="Score", hinge_pt=None):
        # Apply sigmoid to turn into probability
        p = self.sigmoid(score)

        neg_logs =  -np.log(1-p)

        xs, xlabel = score, "Score"
        if x_axis != "Score":
            xs, xlabel = p, x_axis
       
        _= ax.plot(xs, neg_logs, label="- log(1-p)")
        _= ax.set_title("Negative examples")
        _= ax.set_xlabel(xlabel)

        if hinge_pt is not None:
            hinge = self.hinge(score, target=-1, hinge_pt=hinge_pt)
            _= ax.plot(xs, hinge, label="hinge")

        _= ax.legend()

    def plot_log_p(self, x_axis=None, hinge_pt=None):
        fig, axs = plt.subplots(1,2, figsize=(12, 4.5))

        score = np.linspace(-3,+3, num=100)
        _ = self.plot_pos_examples(score, axs[0], x_axis=x_axis, hinge_pt=hinge_pt)
        _ = self.plot_neg_examples(score, axs[1], x_axis=x_axis, hinge_pt=hinge_pt)
        
        fig.tight_layout()

    def plot_hinges(self, hinge_pt=0):
        fig, axs = plt.subplots(1,2, figsize=(12, 4.5))
        score = np.linspace(-3,+3, num=100)
        pos, neg = 1, -1
        hinge_p = np.maximum(0,  pos * (hinge_pt - score) )
        hinge_n = np.maximum(0,  neg * (hinge_pt - score) )

        hinge_p = self.hinge(score, target=+1, hinge_pt=hinge_pt)
        hinge_n = self.hinge(score, target=-1, hinge_pt=hinge_pt)
        
        _= axs[0].plot(score, hinge_p)
        _= axs[0].set_xlabel("Score")
        _= axs[0].set_title("Positive examples")

        _= axs[1].plot(score, hinge_n)
        _= axs[1].set_xlabel("Score")
        _= axs[1].set_title("Negative examples")

        fig.tight_layout()


    # Adapted from external/PythonDataScienceHandbook/notebooks/05.07-Support-Vector-Machines.ipynb

    def val_to_color(self, y):
        cdict = { 0: "red", 1: "green"}
        c = [ cdict[y_val] for y_val in y]

        return c
        
    def make_circles(self, ax=None, plot=False):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(12,6) )
            
        X, y = make_circles(100, factor=.1, noise=.1)

        if plot:
            mask = y > 0
            ax.scatter(X[mask, 0], X[mask, 1], c=self.val_to_color(y[mask]), s=50, label="Positive")
            ax.scatter(X[~mask, 0], X[~mask, 1], c=self.val_to_color(y[~mask]), s=50, label="Negative")
            
            # plt.scatter(X[:, 0], X[:, 1], c=self.val_to_color(y), s=50, cmap='autumn')
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.legend()
            
        return X,y

 
    def plot_svc_decision_function(self, model, ax=None, plot_support=True):
        """
        Plot the decision function for a 2D SVC
        """
        if ax is None:
            ax = plt.gca()
            
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=1, facecolors='none');
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
            

    def circles_linear(self, X, y, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(12,6) )
            
        clf = SVC(kernel='linear').fit(X, y)

        mask = y > 0
        ax.scatter(X[mask, 0], X[mask, 1], c=self.val_to_color(y[mask]), s=50, label="Positive")
        ax.scatter(X[~mask, 0], X[~mask, 1], c=self.val_to_color(y[~mask]), s=50, label="Negative")
        ax.legend()
        
        self.plot_svc_decision_function(clf, ax=ax, plot_support=False);

  
    def plot_3D(self, elev=30, azim=30, X=[], y=[], ax=None, x3label="$x_3$"):
        if ax is None:
            plt.figure(figsize=(12,6))
            ax = plt.subplot(projection='3d')
            
        #ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, s=50, cmap='autumn')
        mask = y > 0
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],  c=self.val_to_color(y[mask]), s=50, label="Positive")
        ax.scatter(X[~mask, 0], X[~mask, 1], X[~mask, 2], c=self.val_to_color(y[~mask]), s=50, label="Negative")
        ax.legend()

        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(x3label, rotation=0)

        return ax

    def circles_radius_transform(self, X):
        r = -(X ** 2).sum(1)

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

    def circles_rbf_transform(self, X):
        r = np.exp( -(X ** 2).sum(1) )

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

    def circles_square_transform(self, X):
        r = np.zeros( X.shape[0] )
        r[ np.all(np.abs(X) <= 0.5, axis=1) ] = 1

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

    def gen_blobs(self, n_samples=200, repeat=None):
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=6)

        if repeat:
            # Imbalance data by replicating one class
            eps = np.array([ X[:,0].min() * .5, X[:,1].min() * .5 ] )
            mask = y != 0
            X = np.concatenate( (X, X[ mask ]) + eps, axis=0)
            y = np.concatenate( (y, y[ mask ]), axis=0)
    
        return X, y

    def svm_equations(self, clf):
        """
        Create string representation of the equation for each separating boundary
        - variables are represented as $x_i$ so are suitable for matplotlib or laTex

        e.g.,
        - c = clf.coefs_[i] are the coefficients of the i-th separating plane
        - b =  clf.intercetp_[i] is the intercept

        -- so eqns[i] ==  sum_{i=0}^{clf.coefs_.shape[0]} { c[i] * "$x_i$" } + b = 0


        Parameters
        ----------
        clf: Classifier

        Returns
        -------
        eqns: array.   eqns[i] is a string representing the equation for boundary i
        """
        coefs_all = clf.coef_

        eqns = []
        for idx in range(0, coefs_all.shape[0]):
            coefs = coefs_all[idx]
            feats = [ "$x_{i:d}$".format(i=i) for i in range(0,len(coefs)) ]

            terms = [ "{co:3.2f} * {ft:s}".format(co=coefs[i], ft=feats[i]) for i in range(0, len(coefs)) ]
            eqn = " + ".join(terms)
            eqn = eqn + " + {x:3.2f} = 0".format(x=clf.intercept_[idx])

            eqns.append(eqn)
            
        return eqns

    def boundary_to_fn(self, clf, boundary_idx, feature_idx):
        """
        Convert eqn of boundary line
        - from form "weighted sum of variables = 0"
        - to   form "x_j = weighted sum of non-j variables"

        Parameters
        ----------
        clf: sklearn object responding to methods coef_ and intercept_

        boundary_idx: Integer.  Index of one boundary in the list of boundaries
        feature_idx:  Integer.  Index of the feature to put on the LHS of the equation

        e.g.,
        - c = clf.coefs_[i] are the coefficients of the i-th separating plane
        - b =  clf.intercept_[i] is the intercept

        -- so eqns[i] ==  sum_{i=0}^{clf.coefs_.shape[0]} { c[i] * "$x_i$" } + b = 0
        --- is equation of the i-th boundary line

        --- for each j in range(0, len(c), we can re-arrange eqns[i] so that x_j is on the LHS
        ---- i.e., transform equation from  "weighted sum of variables = 0" form to "x_j = weightd sum of variables other than x_j" form

        """
        coefs_all = clf.coef_
        intercept_all = clf.intercept_

        # Find parameters of the boundary indexed by boundary_idx
        coefs = coefs_all[boundary_idx]
        xcpt  = intercept_all[boundary_idx]

        # Create the parameters of equation with feature feature_idx on LHS
        j = feature_idx
        
        # Keep the non-j coefficients
        non_j = coefs.tolist()
        del non_j[j]

        # Prepend the intercept
        non_j.insert(0, xcpt)

        # negate all the coefficients
        non_j = -1 * np.array(non_j)

        # Divide by coefs[j]
        non_j = non_j/coefs[j]
                
        return non_j

    def create_kernel_data(self, n_samples=200, classifiers=None):
        # Create some data
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X = X[y != 0, :2]
        y = y[y != 0]

        n_sample = len(X)

        np.random.seed(0)
        order = np.random.permutation(n_sample)
        X = X[order]
        y = y[order].astype(np.float)

        X_train = X[:int(.9 * n_sample)]
        y_train = y[:int(.9 * n_sample)]
        X_test = X[int(.9 * n_sample):]
        y_test = y[int(.9 * n_sample):]

        self.X, self.y = X_train, y_train

        if classifiers is None:
            gamma=1
            C=0.1

            linear_svm = svm.LinearSVC()
            linear_kernel_svm = svm.SVC(kernel="linear", gamma=gamma)
            rbf_kernel_svm = svm.SVC(kernel="rbf", gamma=10)
            poly2_kernel_svm = svm.SVC(kernel="poly", degree=2, gamma=gamma)

            # Pipelines
            feature_map_fourier = RBFSampler(gamma=10, random_state=1)

            fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                                    ("svm", svm.LinearSVC())
                                                   ])

            feature_map_poly2 = PolynomialFeatures(2)
            poly2_approx = pipeline.Pipeline( [ ("feature map", feature_map_poly2),
                                                 ("svm", svm.LinearSVC())
                                              ])

            classifiers =  [ ("linear SVM", linear_kernel_svm),
                            ("linear no transform + SVC", linear_svm),
                            ("poly (d=2) SVM", poly2_kernel_svm),
                            ("poly (d=2) transform + SVC", poly2_approx),
                            ("rbf SVM", rbf_kernel_svm), 
                            ("rbf transform + SVC",fourier_approx_svm)
                            ]

        self.classifiers = classifiers

    def plot_kernel_vs_transform(self, show_margins=True):
        X, y = self.X, self.y
        clh = self.clh
        classifiers = self.classifiers
        
        fig, axs = plt.subplots( math.ceil(len(classifiers)/2.0), 2, figsize=(12, 5 * math.ceil(len(classifiers)/2.0)) )
        axs = np.ravel(axs)

        for i, clf_spec in enumerate(classifiers):
            (clf_name, clf) = clf_spec
            ax = axs[i]

            _= clf.fit(X, y)

            do_scatter=True

            _= ax.axis('tight')
            x_min = X[:, 0].min()
            x_max = X[:, 0].max()
            y_min = X[:, 1].min()
            y_max = X[:, 1].max()

            _= clh.plot_boundary_2(clf, X, y, 
                                   ax=ax, 
                                   xlims=[ (x_min, x_max), (y_min, y_max)], 
                                   cmap=plt.cm.Paired,
                                   show_margins=show_margins,
                                   scatter=do_scatter)

            _= ax.set_title(clf_name)

        fig.tight_layout()

        return fig, axs

class Charts_Helper():
    def __init__(self, save_dir="/tmp", visible=True, **params):
        """
        Class to produce charts (pre-compute rather than build on the fly) to include in notebook

        Parameters
        ----------
        X: ndarray. Two dimensionsal feature
        y: ndarray. One dimensional target

        save_dir: String.  Directory in which charts are created
        visible: Boolean.  Create charts but do/don't display immediately
        """
        self.X, self.y = None, None
        self.save_dir = save_dir

        self.visible = visible

        svmh = SVM_Helper()
        self.svmh = svmh
        
        clh = class_helper.Classification_Helper()
        self.clh = clh

        return

        
    def create_data(self, n_samples=200):
        svmh = self.svmh
        self.n_samples = n_samples
          
        X,y = svmh.gen_blobs(n_samples=n_samples, repeat=False)
        self.X, self.y = X, y
      
    def create_sep_bound(self):
        """
        Chart to illustrate sensitivity of model to examples near margin

        Create chart showing separating boundary
        - Same training examples
        - Same SVC
        - Different value for C
        """
        svmh, clh = self.svmh, self.clh
        X, y = self.X, self.y

        visible = self.visible
        
        # Chart: demonstrate separating boundary
        Cs = np.array([.001, 10])
        fig, axs = plt.subplots(1,2, figsize=(12,6) )

        axs = np.ravel(axs)

        # Train same model/same data, with different values for C
        for i, C in enumerate(Cs):
            ax = axs[i]
            clf = svm.SVC(kernel='linear', C=C)
            _= clf.fit(X, y)
            _= clh.plot_boundary_2(clf, X, y, ax=ax, scatter=True,
                                   cmap=ListedColormap(('navajowhite', 'darkkhaki')),
                                   show_margins=True, margins=[-.001, 0, .001]
                                  )

            eqns = svmh.svm_equations(clf)
            _= ax.set_title("C={c:3.3f}: \n {e:s}".format(c=C, e=eqns[0]))

        if not visible:
            plt.close(fig)
            
        return fig, axs

    def create_sens(self):
        """
        Create figure showing sensitivity of a classifier to new examples far fro boundary
        """
        
        svmh, clh = self.svmh, self.clh
        
        X, y = self.X, self.y
        visible = self.visible

        # Create the classifier models
        clf_log = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, C=10)
        clf_svm = svm.SVC(kernel='linear', C=10)

        # Create "small" dataset
        n_samples = 60
        X_orig, y_orig = svmh.gen_blobs(n_samples=n_samples, repeat=False)

        Cs = np.array([.001, 1000])
        fig, axs = plt.subplots(2,2, figsize=(12,9) )

        clf_c = clf_svm

        # Create one row of figure for each model
        for i, clf_spec in enumerate( [ (clf_log, "Logistic"), (clf_svm, "SVC") ]):
            clf, title = clf_spec
            X, y = X_orig, y_orig

            # Run the model twice:
            # - Once with original examples
            # - Second time after adding examples to original
            for j in [0,1]:
                ax = axs[i,j]

                # Plot model results/separating boundary
                _= clf.fit(X, y)
                _= clh.plot_boundary_2(clf, X, y, ax=ax, scatter=True,
                                       cmap=ListedColormap(('navajowhite', 'darkkhaki')),
                                       show_margins=True, margins=[-.001, 0, .001]
                                      )

                # Compute the SVC maring
                # - need to ensure that newly added examples are outside the margin
                margin = 1 / np.sqrt(np.sum(clf.coef_[0] ** 2))

                # Add a bunch of class=1 examples on the correct side of the original boundary
                w_2 = svmh.boundary_to_fn(clf, 0, 1)

                # Vectors of coefficients (for intercept and x_0)
                X0_a = np.array( [ [1,10] , [1,11] ])

                # Compute corresponding x_1 values on the boundary
                X1_a = np.dot(X0_a, w_2)

                # Move the x_1 below the boundary (make sure it is **greater than ** margin)
                X1_a = X1_a - (margin  + .001) # 1.55
                X_add  = np.concatenate( (X0_a[:,-1].reshape(-1,1), X1_a.reshape(-1,1)), axis=1)

                # Replicate the examples to boost their influence
                X_add = np.repeat(X_add, 20, axis=0)

                # The class of all new examples is 1
                y_add = np.repeat( [1], X_add.shape[0])

                # Add the new bunch of examples to the original
                X = np.concatenate( (X, X_add), axis=0)
                y = np.concatenate( (y, y_add), axis=0) 

                eqns = svmh.svm_equations(clf)
                _= ax.set_title("{m:s}: {e:s}".format(m=title, e=eqns[0]))

        fig.tight_layout()

        if not visible:
            plt.close(fig)
            
        return fig, axs

    def create_margin(self, Cs=[10, .1]):
        svmh, clh = self.svmh, self.clh
        X, y = self.X, self.y

        visible = self.visible

        Cs = np.array(Cs)
            
        fig, axs = plt.subplots(1, Cs.shape[0], figsize=(12,6) )

        axs = np.ravel(axs)
        for i, C in enumerate(Cs):
            ax = axs[i]
            clf = svm.SVC(kernel='linear', C=C)
            _= clf.fit(X, y)
            _= clh.plot_boundary_2(clf, X, y, ax=ax, scatter=True,
                                   cmap=ListedColormap(('navajowhite', 'darkkhaki')),
                                   show_margins=True
                                  )

            eqns = svmh.svm_equations(clf)
            _= ax.set_title("C={c:3.3f}: \n {e:s}".format(c=C, e=eqns[0]))

        if not visible:
            plt.close(fig)

        return fig, axs
    
    def create_charts(self):
        save_dir = self.save_dir
        
        _= self.create_data()

        print("Saving to directory: ", save_dir)
        
        print("Create Boundary chart")
        fig, ax = self.create_sep_bound()
        bound_file = os.path.join(save_dir, "svm_sep_boundary.png")
        fig.savefig(bound_file)

        print("Create Sens chart")
        fig, axs = self.create_sens()
        sens_file = os.path.join(save_dir, "svm_sens.png")
        fig.savefig(sens_file)

        print("Create margin chart")
        fig, axs = self.create_margin(Cs=[ .1, 10])
        margin_file = os.path.join(save_dir, "svm_margin.png")
        fig.savefig(margin_file)
        
        print("Done")
        
        return { "boundary": bound_file,
                 "sensitivity": sens_file,
                 "margin": margin_file
                 }

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, validation_curve

import os
import tempfile
import functools

from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

import pdb

class Charts_Helper():
    def __init__(self, save_dir=None, visible=True, **params):
        # Default for save_dir is the system's temp dir
        if save_dir == None:
            save_dir = tempfile.gettempdir()

        self.X, self.y = None, None
        self.save_dir = save_dir
            
        # Ensure directory exists
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
        self.visible = visible

        return

    def create_data(self, n_samples=30):
        # Load the diabetes dataset
        X, y = datasets.load_diabetes(return_X_y=True)

        self.X, self.y = X, y
    
    def create_fit(self, n_samples=30):
        X, y = self.X, self.y
        save_dir = self.save_dir
        visible = self.visible
        
        # Use only one feature
        X = X[:, np.newaxis, 2]

        # Split the data into training/testing sets
        X_train = X[:-n_samples]
        X_test = X[-n_samples:]

        # Split the targets into training/testing sets
        y_train = y[:-n_samples]
        y_test = y[-n_samples:]

        # Create linear regression object
        regr = linear_model.LinearRegression()
        self.regr = regr

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Plot train
        fig, ax = plt.subplots(1,1, figsize=(5,3))
        _= ax.scatter(X_train, y_train,  color='black')
        _= ax.set_xlabel("x")
        _= ax.set_ylabel("y")

        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())

        scatter_file = os.path.join(save_dir, "regr_scatter.jpg")
        fig.savefig(scatter_file)
        
        if not visible:
            plt.close(fig)
            
        # Plot fit
        fig1, ax1 = plt.subplots(1,1, figsize=(5,3))
        _= ax1.scatter(X_train, y_train,  color='black')
        _= ax1.set_xlabel("x")
        _= ax1.set_ylabel("y")

        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_major_formatter(plt.NullFormatter())

        y_train_pred = regr.predict(X_train)

        sort_idx = X_train[:, 0].argsort()
        _= ax1.plot( X_train[sort_idx, 0], y_train_pred[sort_idx], color="red", linewidth=3)

        fit_file = os.path.join(save_dir, "regr_scatter_with_fit.jpg")
        fig.savefig(fit_file)
        
        if not visible:
            plt.close(fig1)
            
        return ax, ax1

class Recipe_Helper():
    def __init__(self, v=4,  a=0, **params):
        self.v, self.a = v, a
        self.X, self.y = None, None
        
        return

    def gen_data(self, num=30):
        """
        Generate a dataset of independent (X) an dependent (Y)

        Parameters
        -----------
        num: Integer.  The number of observations

        Returns
        --------
        (X,y): a tuple consisting of X and y.  Both are ndarrays
        """

        # Make up data on house prices, based on area
        # Create a range of lengths and widths, and from that: areas
        min_length, max_length = 20, 50
        min_width,  max_width  = 10, 40
        length = np.arange(min_length, max_length, (max_length - min_length)/num)
        width  = np.arange(min_width, max_width, (max_width - min_width)/num)

        area = length * width 

        # Create a not-quite-linear relationship between area and price
        price = 100e3 + 100 * area + .05* area**2

        # Scale price down to 000's for readability
        price = price/1000.0
        
        self.X, self.y = area, price
        self.X, self.y = self.X.reshape(-1,1), self.y.reshape(-1,1)
        
        return self.X, self.y


    def gen_data_v0(self, num=30):
        """
        Generate a dataset of independent (X) an dependent (Y)

        Parameters
        -----------
        num: Integer.  The number of observations

        Returns
        --------
        (X,y): a tuple consisting of X and y.  Both are ndarrays
        """
        v, a = self.v, self.a
        rng = np.random.RandomState(42)


        # X = num * rng.uniform(size=num)
        X = num * rng.normal(size=num)
        # X = X - X.min()

        X = X.reshape(-1,1)

        e = (v + a*X)
        y = v * X #  +  e * rng.uniform(-1,1, size=(num,1))

        a_term =  0.5 * a * (X**2)
        y = y + a_term

        self.X, self.y = X, y

        return X,y

    def gen_plot(self, X,y, xlabel, ylabel):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)


        _ = ax.scatter(X, y, color="red")


        _ = ax.set_xlabel(xlabel)
        _ = ax.set_ylabel(ylabel)

        return fig, ax


    def split(self, X,y, shuffle=True, pct=.80, seed=42):
        """
        Split the X and y datasets into two pieces (train and test)

        Parameters
        ----------
        X, y: ndarray.

        pct: Float.  Fraction (between 0 and 1) of data to assign to train
        seed: Float.  Seed for the random number generator

        Returns
        -------
        Tuple of length 4: X_train, X_test, y_train, y_test
        """
        # Random seed
        rng = np.random.RandomState(42)

        # Number of observations
        num = y.shape[0]

        # Enumerate index of each data point  
        idxs = list( range(0, num))

        # Shuffle indices
        if(shuffle):
            rng.shuffle(idxs)

        # How many observations for training ?
        split_idx = int( num * pct)

        # Split X and Y into train and test sets
        X_train, y_train = X[ idxs[:split_idx] ] , y[ idxs[:split_idx] ]
        X_test,  y_test  = X[ idxs[split_idx:] ],  y[ idxs[split_idx:] ]

        return X_train, X_test, y_train, y_test

    def plot_fit(self, X, y, ax=None,  on_idx=0):
        """
        Plot the fit

        Parameters
        ----------
        X: ndarray of features
        y: ndarray of targets
        ax: a matplotlib axes pbject (matplotlib.axes._subplots.AxesSubplot)

        Optional
        --------
        on_idx: Integer.  Which column of X to use for the horizontal axis of the plot

        """
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(1,1,1)
            
        sort_idx = X[:, on_idx].argsort()
        X_sorted = X[ sort_idx,:]
        y_sorted = y[ sort_idx,:]

        _ = ax.plot(X_sorted[:, on_idx] , y_sorted, color="red")

        return ax
    
    def transform(self, X):
        """
        Add a column to X with squared values

        Parameters
        ----------
        X: ndarray of features
        """
        X_p2 = np.concatenate( [X, X **2], axis=1)
        return X_p2

    def run_regress(self, X,y, model=None, run_transforms=False, plot_train=True,  xlabel=None, ylabel=None, print_summary=True):
        """
        Do the full pipeline of the regression of y on X

        Parameters
        ----------
        X: ndarray of features
        y: ndarray of targets

        Optional
        --------
        model: an sklearn model. If it is None, a linear_model will be used
        runTransforms: Boolean.  If True, run additional data transformations to create new features
        """
        self.X, self.y = X, y

        if model is None:
            # Create linear regression object
            regr = linear_model.LinearRegression()
        else:
            regr = model

        # Split into train, test
        X_train, X_test, y_train, y_test = self.split(X,y)

        # Transform X's
        if (run_transforms):
            X_train = self.transform(X_train)
            X_test  = self.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        

        self.model = regr
        
        # Train the model using the training sets
        _ = regr.fit(X_train, y_train)

        # The coefficients
        if print_summary and  hasattr(regr, "intercept_") and hasattr(regr, "coef_"):
            print('Coefficients: \n', regr.intercept_, regr.coef_)

        # Lots of predictions: predict on entire test set
        y_pred = regr.predict(X_test)

        # Explained variance score: 1 is perfect prediction
        rmse_test = np.sqrt( mean_squared_error(y_test,  y_pred))

        if print_summary:
            print("\n")
            print("R-squared (test): {:.2f}".format(r2_score(y_test, y_pred)) )
            print("Root Mean squared error (test): {:.2f}".format( rmse_test ) )

        y_pred_train = regr.predict(X_train)

        rmse_train = np.sqrt( mean_squared_error(y_train,  y_pred_train))

        if print_summary:
            print("\n")
            print("R-squared (train): {:.2f}".format(r2_score(y_train, y_pred_train)) )
            print("Root Mean squared error (train): {:.2f}".format( rmse_train ) )

        # Plot predicted ylabel (red) and true label (black)
        num_plots = 2
        fig, axs = plt.subplots(1,num_plots, figsize=(12,4))
        
        _ = axs[0].scatter(X_test[:,0], y_test, color='black')
        _ = axs[0].scatter(X_test[:,0], y_pred, color="red")

        _= self.plot_fit(X_test, y_pred, ax=axs[0], on_idx=0)
        
        if xlabel is not None:
            _ = axs[0].set_xlabel(xlabel)

        if ylabel is not None:
            _ = axs[0].set_ylabel(ylabel)

        axs[0].set_title("Test (RMSE={e:3.2f})".format(e=rmse_test))

        
        # Plot train
        if plot_train:
            _ = axs[1].scatter(X_train[:,0], y_train, color='black')
            _ = axs[1].scatter(X_train[:,0], y_pred_train, color="red")

            self. plot_fit(X_train, y_pred_train, ax=axs[1], on_idx=0)
            if xlabel is not None:
                _ = axs[1].set_xlabel(xlabel)

            if ylabel is not None:
                _ = axs[1].set_ylabel(ylabel)

            axs[1].set_title("Train (RMSE={e:3.2f})".format(e=rmse_train))
            
        return fig, axs

    def regress_with_error(self, X,y, model=None, run_transforms=False, plot_train=True,  xlabel=None, ylabel=None):
        # Run the regression.  Sets attributes of self
        self.run_regress(X, y, model, run_transforms, plot_train=plot_train, xlabel=xlabel, ylabel=ylabel)

        # Extract the results of running the split and regression steps
        X_train, X_test, y_train, y_test, regr = self.X_train, self.X_test, self.y_train, self.y_test, self.model

        y_pred = regr.predict(X_test)
        y_pred_train = regr.predict(X_train)

        fig, axs = plt.subplots(1,2, figsize=(12,4))

        # Plots for both test and train datasets
        for i, spec in enumerate( [ ("test", X_test, y_test, y_pred), ("train", X_train, y_train, y_pred_train) ] ):

            label, x, target, pred = spec
            ax = axs[i]
            _= ax.scatter(x, target - pred, label="Error")
            # _= ax.bar(x.reshape(-1), (target - pred).reshape(-1), label="Error")
            _= ax.set_xlabel("Error")
            _= ax.set_ylabel(ylabel)
            _= ax.set_xlabel(xlabel)
            _= ax.set_title(label + " Error")
            _= ax.legend()

        return fig, axs
        
    def plot_resid(self, X, y, y_pred):
        resid_curve = y - y_pred
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.scatter(X, resid_curve)
        _ = ax.set_xlabel(xlabel)
        _ = ax.set_ylabel("Residual")

    def compare_regress(self, X,y, model=None, plot_train=False,  xlabel=None, ylabel=None, print_summary=True, visible=False):
        """
        Compare the linear model to one with second order polynomial
        """

        # Run the linear model
        fig_lin, axs_lin = self.run_regress(X, y, xlabel=xlabel, ylabel=ylabel, plot_train=plot_train, print_summary=False)

        # Run the second order model
        fig_curv, axs_curv = self.run_regress(X, y, run_transforms=True, xlabel=xlabel, ylabel=ylabel, plot_train=plot_train, print_summary=False)

        # Get rid of grid element for training is not plot_train
        if not plot_train:
            axs_lin[-1].remove()
            axs_curv[-1].remove()
        
        if not visible:
            plt.close(fig_lin)
            plt.close(fig_curv)
            
        return { "linear": (fig_lin, axs_lin),
                 "second order": (fig_curv, axs_curv)
                 }
    
                

class Bias_vs_Variance_Helper():
    def __init__(self, true_fun, pipe, **params):
        self.true_fun = true_fun
        self.pipe = pipe
        self.X, self.y = None, None

        return


    def create_data(self,  n_samples=30):
        true_fun = self.true_fun
        np.random.seed(0)
        
        X = np.sort(np.random.rand(n_samples))

        y = true_fun(X) + np.random.randn(n_samples) * 0.1

        self.X, self.y = X, y
        return X, y

    def plot_degrees(self, degrees=[1,4,15]):
        X, y = self.X, self.y
        true_fun = self.true_fun
        pipe = self.pipe
        
        if X is None or y is None:
            print("X, y need to be initialized")
            return None
        
        fig, axs = plt.subplots(1, len(degrees), figsize=(12,6))
        axs = list(axs)

        for i, degree in enumerate(degrees):
            # Create pipeline with fixed polynomial degree
            polynomial_features = PolynomialFeatures(degree, include_bias=False)
            linear_regression = linear_model.LinearRegression(degree)
            pipe = pipeline.Pipeline([("polynomial_features", polynomial_features),
                                      ("linear_regression", linear_regression)])

            ax = axs[i]
            
            _= pipe.fit(X[:, np.newaxis], y)

            # Evaluate the models using crossvalidation
            scores = cross_val_score(pipe, X[:, np.newaxis], y,
                                     scoring="neg_mean_squared_error", cv=10)

            X_test = np.linspace(0, 1, 100)
            _= ax.plot(X_test, pipe.predict(X_test[:, np.newaxis]), label="Model")
            _= ax.plot(X_test, true_fun(X_test), label="True function")
            _= ax.scatter(X, y, edgecolor='b', s=20, label="Samples")
            _= ax.set_xlabel("x")
            _= ax.set_ylabel("y")
            _= ax.set_xlim((0, 1))
            _= ax.set_ylim((-2, 2))
            _= ax.legend(loc="best")

            test_mse, test_std = - scores.mean(), scores.std()
            fmt = "{:3.2f}" if test_mse < 1000 else "{:.2e}"
            _= ax.set_title(("Degree {}\nTest MSE = " + fmt + " (+/- " + fmt +")").format(
                degree, test_mse, test_std))
        fig.tight_layout()

        return fig, axs

    def plot_validation_curve(self, degrees=[1,4,15]):
        X, y = self.X, self.y
        pipe = self.pipe

        # Compute train and test scores, for each degree in degrees
        param_range = degrees
        train_scores, test_scores = validation_curve(
            pipe, X[:, np.newaxis], y, param_name="polynomial_features__degree", param_range=param_range,
            scoring="neg_mean_squared_error", n_jobs=1, cv=10)

        # Average.StdDev of scores for each degree
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Scores are NEGATIVE errors, convert to error
        mse_train, mse_test = -train_scores_mean, - test_scores_mean
        
        # Plot (vs degree) Training and Test scores
        fig, ax = plt.subplots(1,1, figsize=(12,6))

        _= ax.set_title("Train/Test MSE")
        _= ax.set_xlabel(r"degree")
        _= ax.set_ylabel("Train Score")
        _= ax.set_ylim(mse_train.min(), mse_train.max())
        lw = 2
        _= ax.plot(param_range, mse_train, label="Training score",
                     color="darkorange", lw=lw)

        _= ax.legend()

        axr = ax.twinx()

        #mse_test = np.maximum(mse_test, 1)
        mse_test_max_plot = 1
        mse_test2 = np.minimum(mse_test,1)

        _= axr.plot(param_range, mse_test2, label="Cross-validation score",
                     color="navy", lw=lw)
        _= axr.set_ylabel("Test Score")
        _= axr.set_ylim(mse_test.min(), mse_test_max_plot) 

        _= axr.legend()

        return fig, ax

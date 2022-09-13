import time
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

import pdb

class MNIST_Helper():
    def __init__(self, save_dir="/tmp", visible=True, random_seed=42, **params):
        self.X, self.y = None, None
        self.random_seed = random_seed
        self.visible = visible
        return

    def fetch_mnist_784(self, cache=True):
        """
        Fetch MNIST X, y datasets

        If not caches locally, obtain from web site, and then cache

        Returns
        -------
        X: ndarray, number of images * 784
        - each imaage is 784 pixels: (28,28) grid
        y:labels
        """
        # The fetch from the remote site is SLOW b/c the data is so big
        # Try getting it from a local cache
        cache_dir = "cache/mnist_784"
        (X_file, y_file) = [ "{c}/{f}.npy".format(c=cache_dir, f=fn) for fn in ["X", "y"] ]

        if cache and os.path.isfile(X_file) and os.path.isfile(y_file):
            print("Retrieving MNIST_784 from cache")
            X = np.load(X_file, allow_pickle=True)
            y = np.load(y_file, allow_pickle=True)
        else:
            print("Retrieving MNIST_784 from remote")
            # Load data from hiittps://www.openml.org/d/554
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

            # Cache it !
            os.makedirs(cache_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)

        self.X, self.y = X, y
        return X,y

    def setup(self):
        # Derived from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html

        # Author: Arthur Mensch <arthur.mensch@m4x.org>
        # License: BSD 3 clause

        # Turn down for faster convergence
        train_samples = 5000

        # Fetch the data
        if self.X is not None and self.y is not None:
            X, y = self.X, self.y

        else:
            X, y = self.fetch_mnist_784()

        random_seed = self.random_seed
        
        random_state = check_random_state(random_seed)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_samples, test_size=10000,
            random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = \
        X_train, y_train, X_test, y_test, scaler

    def visualize(self, X=None, y=None, save_file=None):
        """
        Plot a subset of the digits given by X

        Parameters
        ----------
        X: ndarray (num_images * 784)

        """
        if X is None:
            X = self.X_train


        # Plot the first num_rows * num_cols images in X
        (num_rows, num_cols) = (5, 5)
        fig = plt.figure(figsize=(10,10))

        # Plot each image
        for i in range(0, num_rows * num_cols):
            # Reshape image to 2D
            img = X[i].reshape(28, 28)

            ax  = fig.add_subplot(num_rows, num_cols, i+1)
            _ = ax.set_axis_off()

            _ = plt.imshow(img, cmap="gray")

        # Save image in file
        if save_file is not None:
            fig.savefig(save_file)

        visible = self.visible
        
        if not visible:
            plt.close(fig)
            
        return fig

    def fit(self, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X_train

        if y_train is None:
            y_train = self.y_train
            
        train_samples = X_train.shape[0]

        # Turn up tolerance for faster convergence
        clf = LogisticRegression(C=50. / train_samples,  # n.b. C is 1/(regularization penalty)
                                 multi_class='multinomial',
                                 # penalty='l1',   # n.b., "l1" loss: sparsity (number of non-zero) >> "l2" loss (dafault)
                                 solver='saga', tol=0.1)

        t0 = time.time()

        # Fit the model
        _= clf.fit(X_train, y_train)

        run_time = time.time() - t0
        # print('Example run in %.3f s' % run_time)

        self.clf = clf
        return clf

    def plot_coeff(self):
        clf = self.clf

        fig = plt.figure(figsize=(10, 8))
        coef = clf.coef_.copy()

        (num_rows, num_cols) = (2,5)

        scale = np.abs(coef).max()
        for i in range(10):
            ax = fig.add_subplot(num_rows, num_cols, i+1)

            # Show the coefficients for digit i
            # Reshape it from (784,) to (28, 28) so can interpret it
            _ = ax.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                           cmap="gray", #plt.cm.RdBu, 
                           vmin=-scale, vmax=scale)

            _ = ax.set_xticks(())
            _ = ax.set_yticks(())
            _ = ax.set_xlabel('Class %i' % i)

        _= fig.suptitle('Parameters for...')


        return fig, ax


    def create_confusion_matrix(self, expected=None, predicted=None):
        if expected is None:
            expected  = self.y_test

        if predicted is None:
            predicted = self.clf.predict(self.X_test)

        confusion_mat = metrics.confusion_matrix(expected, predicted)

        self.confusion_mat = confusion_mat

        return confusion_mat

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            # Normalize by row sums
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around( 100 * cm_pct, decimals=0).astype(int)

            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # Plot coordinate system has origin in upper left corner
            # -  coordinates are (horizontal offset, vertical offset)
            # -  so cm[i,j] should appear in plot coordinate (j,i)
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def plot_problem_digits(self, problem_digits, wrong_class=None, num_cols=5, expected=None, predicted=None):

        """
        Plot mis-classified digits

        Parameters
        ----------
        problem_digits: List of characters.  Each element of list is a digit
        wrong_class: Character.
        - show only examples that are mis-classified as wrong_class

        num_cols: Integer.  Number of misclassified examples to show for each problem digit

        For each digit T in list: find the test set examples where true label is T but is classified otherwise.
        Plot the mis-classified test examples
        """
        if expected is None:
            expected  = self.y_test

        if predicted is None:
            predicted = self.clf.predict(self.X_test)

        X_test = self.X_test
        
        # Dimensions of plot grid
        num_rows = len(problem_digits)
        fig = plt.figure(figsize=(2*num_cols, 3*num_rows))

        misclassified = {}
        
        # Plot examples for each problem digit
        for i, digit in enumerate(problem_digits): 
            # Find the mis-classified test obsevations for this digit
            # Which misclassified examples do we want ?
            # -- All
            # -- Only those classified as wrong_class
            if (wrong_class is not None) and (wrong_class != digit):
                wrong_predict = (predicted == wrong_class)
            else:
                wrong_predict = (predicted != expected)
                
            # mask = (expected == digit) & (expected != predicted)
            mask = (expected == digit) & wrong_predict
            
            X_misclassified = X_test[mask]
            y_misclassified = predicted[mask]

            # Save the mis-classified for future examination
            misclassified[digit] = X_misclassified
            
            num_misclassified = X_misclassified.shape[0]

            # Plot the mis-classified instance of digit
            plot_num = num_cols * i
            for j in range(0, min(num_cols, num_misclassified)):
                # Get the X, y for the mis-classified image
                img = X_misclassified[j].reshape(28,28)
                pred =   y_misclassified[j]

                # Plot the image
                ax = fig.add_subplot(num_rows, num_cols, plot_num + j +1)
                _ = ax.set_axis_off()

                _ = plt.imshow(img, cmap="gray")
                _ = ax.set_title("Pred: {c:s},\n Class: {t:s}".format(c=pred, t=digit))

        self.misclassified = misclassified

        plt.tight_layout()


    def predict_with_probs(self, X, classes):
        predicted = self.clf.predict(X)
        probs     = self.clf.predict_proba(X)

        fig, axs = plt.subplots( X.shape[0], 2, figsize=(12, 3* X.shape[0]) )

        
        idxs = range(len(classes))
        i = 0
        for pred, prob in zip(predicted, probs):
            ax = axs[i]
            _ = ax[0].imshow(X[i].reshape(28, 28),
                             interpolation='nearest',
                             cmap="gray"
                             )
            _ = ax[0].set_axis_off()
            _= ax[0].set_title("Predict: " + pred)

            _= ax[1].bar(idxs,   prob )
            _= ax[1].set_xticks(classes)
            _= ax[1].set_title("Probabilies")

            i += 1

        plt.tight_layout()

    def make_binary(self, digit, y_train=None, y_test=None):
        """
        Turn multinomial target into binary target

        Parameters
        ----------
        digit: Character.  Value of digit on which to make target binary: "Is digit"/"Is NOT digit"
        y_train, y_test: ndarrays.  Train/test target values

        Returns
        -------
        Tuple (y_train, y_test)
        """
        if y_train is None:
            y_train = self.y_train

        if y_test is None:
            y_test = self.y_train
            
        y_train_d = ( y_train == digit)
        y_test_d  = ( y_test  == digit)

        return  y_train_d, y_test_d


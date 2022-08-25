import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

from matplotlib.colors import ListedColormap

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools
import itertools

from ipywidgets import interact, fixed

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pdb

class Classification_Helper():
    def __init__(self, **params):
        self.Debug = False
        return

    def plot_boundary_2(self, clf,  X, y,
                        ax=None, cmap=plt.cm.RdYlBu,
                        labels=None,
                        scatter=True,
                        xlims=None,
                        feature_names=None,
                        alpha=0.75,
                        show_legend=True,
                        show_margins=False, show_vectors=False,
                        margins=[-1, 0, 1]
                        ):

        """
        Plot
        - scatter plot of examples
        - decision boundaries between classes

        for arbitrary (?) classifier and examples with exactly 2 features

        Parameters
        ----------
        clf: sklearn classifier object
        X: ndarray of examples, len(X.shape[1]) == 2
        - X can have only 2 features
        y: ndarray Integer target values

        Optional
        --------
        ax: matplotlib Axes in which to plot

        cmap: matplotlib ColorMap
        - colors of classes, boundaries

        label: array
        - label (on plot) of point in class

        xlims: array of length 2
        - xlims[i][0], xlims[i][1]: minimum and maximum value of feature x_i
        - for i in [0,1]

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))

        if labels is None:
            labels=range(0, np.unique(y).shape[0])

        if feature_names is None:
            feature_names = [ "$x_{idx:d}$".format(idx=i+1) for i in range(0, 2) ]
            
        # Set limits for each feature
        if xlims:
            x0_min, x0_max = xlims[0]
            x1_min, x1_max = xlims[1]
        else:
            x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Grid of feature_0, feature_1 values
        X0, X1 = np.meshgrid(np.arange(start = x0_min,
                                       stop  = x0_max,
                                       step = 0.01),
                             np.arange(start = x1_min,
                                       stop = x1_max,
                                       step = 0.01)
                             )

        # Predict at each element (x_0, x_1) of the grid
        xy = np.array([X0.ravel(), X1.ravel()]).T
        predicts = clf.predict(xy)
        predicts = predicts.T.reshape(X0.shape)
        
        # Plot the predictions, color coded by label (y)
        _= ax.contourf(X0, X1, predicts,
                       alpha = alpha,
                       cmap=cmap
                       )

        # Set limits of horizontal and vertical axes
        _= ax.set_xlim(X0.min(), X0.max())
        _= ax.set_ylim(X1.min(), X1.max())

        colors= [ "red", "green", "blue", "yellow", "white", "black", "gray", "orange", "indigo", "violet"]
        
        # Scatter plot, color coded by class label (y)
        # Set size of marker
        # - if we draw support vectors: we need to place a slighlty larger marker on top of a few points
        s=30
        if scatter:
            for i, j in enumerate(np.unique(y)):
                _= ax.scatter(X[y == j, 0], X[y == j, 1],
                              s=s,
                              c = colors[i], 
                              cmap=cmap,
                              label = labels[i],
                              edgecolor='black' # Outline of point
                )
            if show_legend:
                _= ax.legend()
                
        _= ax.set_xlabel(feature_names[0])
        _= ax.set_ylabel(feature_names[1])

        # The following apply only when clf is an SVC
        # plot decision boundary and margins
        has_fn = hasattr(clf, "decision_function") or hasattr(clf, "predict_proba")
        if has_fn and show_margins:
            # NOTE: args are X1, X0 NOT X0, X1
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(xy)
            else:
                Z = clf.predict_proba(xy)[:, 1]

            Z = Z.reshape(X0.shape)
            
            _= ax.contour(X0, X1, Z,
                          colors='k',
                          levels=margins,
                          alpha=0.5,
                          linestyles=['--', '-', '--']
            )

            has_fn = hasattr(clf, "support_vectors")
        if has_fn and show_vectors:
            # plot support vectors
            _= ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=s*2.5, # Marker should be bigger than original
                          linewidth=1, facecolors='none', edgecolors='k')


        # Return ax (in case it was created by this method)
        return ax
    
    def load_digits(self):
        """
        Load the MNIST (small) dataset

        Returns
        -------
        X_digits, y_digits: ndarrays
        - X_digits[i]: array of pixels
        - y_digits[i]: label of the image (i.e, which digit image is)
        """
        
        digits = datasets.load_digits()
        X_digits = digits.data / digits.data.max()
        y_digits = digits.target

        self.size = int( np.sqrt( X_digits.shape[1] )  )
        return X_digits, y_digits

    def split_digits(self, X, y, random_state=42):
        """
        Split X,y into train and test

        Parameters
        ----------
        X, y: ndarrays
        random_state: integer
        - initialization of random seed

        Returns
        -------
        X_train, X_test, y_train, y_test
        - X_train, X_test: partition of X
        - y_train, y_test: partition of  y, corresponding to the partition of X
        """
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=100,
            shuffle=True, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def plot_digits(self, X_digits, y_digits, digits_per_row=10):
        """
        Plot images of digits and their labels

        Parameters
        ----------
        X_digits, y_digits: ndarrays
        - X_digits[i]: array of pixels of an image
        - y_digits[i]: int.  The label of the image
        """
        
        digits = range(y_digits.min(), y_digits.max() +1)

        (num_rows, num_cols) = (len(digits) , digits_per_row)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12,8))
        
        num_shown = 0

        # Plot a sample of each digit
        for row, digit in enumerate(digits):
            this_digits = X_digits[ y_digits == digit ]
            imgs = [ img.reshape(self.size, self.size) for img in this_digits[:num_cols, :] ]
            for col, img in enumerate(imgs):
                ax = axs[row,col]
                _ = ax.set_axis_off()

                _ = ax.imshow(img, cmap = mpl.cm.binary)

        return fig, axs
    
    def fit_digits(self, X_digits, y_digits):
        """
        Fit a classifier

        Parameters
        ----------
        X_digits, y_digits: ndarrays
        - X_digits[i]: array of pixels of image of a digit
        - y_digits[i]: int.  Label of the image.

        Returns
        -------
        X_train, X_test, y_train, y_test, models
        - X_train, X_test, y_train, y_test: ndarrays (see split_digits)
        - models: Dict
        -  models["knn"]: the sklearn KNN model, fit to X_train, y_train
        """
        X_train, X_test, y_train, y_test = self.split_digits(X_digits, y_digits)
        knn = neighbors.KNeighborsClassifier()
        logistic = linear_model.LogisticRegression(solver='lbfgs',
                                                   max_iter=1000,
                                                   multi_class='multinomial')

        print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
        print('LogisticRegression score: %f'
              % logistic.fit(X_train, y_train).score(X_test, y_test))

        models = { "knn": knn,
                   "lr" : logistic
                   }

        return  X_train, X_test, y_train, y_test, models

    def plot_digit(self, img, ax=None):
        """
        Plot an image

        Parameters
        ----------
        img: ndarray (one dimension) of pixels; image assumed to be square
        """

        # Create an axis if none given
        if ax is None:
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, num_rows *1.5))
        _= ax.imshow( img.reshape(self.size, self.size), cmap = mpl.cm.binary)
        _= ax.set_axis_off()

        return ax
        
    def predict_digits(self, model, X_digits, y_digits):
        """
        Create predictions, given a model

        Parameters
        ----------
        model: fitted sklearn model
        X_digits, y_digits: ndarrays
        - X_digits[i]: image of a digit to predict
        - y_digits[i]: correct label of image
        """
        preds = model.predict(X_digits)

        digits_per_row = 5
        num_rows = X_digits.shape[0] // digits_per_row
        num_rows += num_rows + 1 if (0 != X_digits.shape[0] % digits_per_row) else 0
        num_cols = digits_per_row
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows *1.5))

        plt.subplots_adjust(hspace=0.32)

        # Plot each prediction
        for i in range( preds.shape[0]):
            img, pred, target = X_digits[i], preds[i], y_digits[i]
            row, col = i // num_cols, i % num_cols
            ax = axs[row,col]

            # Plot the digit
            self.plot_digit(img, ax)

            # Set title to indicate whether prediction was correct
            if pred == target:
                label = "Correct {dig:d}".format(dig=pred)
            else:
                label = "Incorrect: Predict {p:d}, is {t:d}".format(p=pred, t=target)
            ax.set_title(label)

        return fig, axs


    def plot_confusion_matrix(self,
                              cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues,
                              fig=None, ax=None
                              ):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if title is None:
            title = 'Confusion matrix'
            if normalize:
                title = title + "(%)"

        if normalize:
            # Normalize by row sums
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around( 100 * cm_pct, decimals=0).astype(int)

            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(classes)
        ax.tick_params(axis='x', labelrotation=45)

        #ax.set_yticks(classes)

        # fmt = '.0%' if normalize else 'd'
        fmt = 'd'
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # Plot coordinate system has origin in upper left corner
            # -  coordinates are (horizontal offset, vertical offset)
            # -  so cm[i,j] should appear in plot coordinate (j,i)
            _= ax.text(j, i, format(cm[i, j], fmt),
                       horizontalalignment="center",
                       verticalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        # plt.tight_layout()

        return fig, ax

    def plot_attrs(self, df, attrs, attr_type="Cat", normalize=True, plot=True):
        """
        Plot/print the distribution of one or more attributes of DataFrame

        Parameters
        ----------
        df: DataFrame
        attrs: List of attributes (i.e., column names)

        Optional
        --------
        attr_type: String; 
          "Cat" to indicate that the attributes in attrs are Categorical (so use value_counts)
          Otherwise: the attributes must be numeric columns (so use histogram)
        """
        num_attrs = len(attrs)
        ncols=2
        nrows = max(1,round(num_attrs/ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, num_attrs*2))

        # Make sure axes is an array (special case when num_attrs==1)
        if num_attrs == 1:
            axes =np.array( [ axes ])

        for i, attr in enumerate(attrs):
            if attr_type == "Cat":
                alpha_bar_chart = 0.55
                plot_data = df.loc[:, attr ].value_counts(normalize=normalize).sort_index()

                args = { "kind":"bar" } #, "alpha":alpha_bar_chart}
                kind="bar"
            else:
                plot_data = df.loc[:, [attr] ]

                args = { "kind":"hist"}
                if normalize:
                    args["density"] = True
                kind="hist"

            if plot:
                ax = axes.flatten()[i]
                _ = plot_data.plot(title=attr, ax=ax, **args)
                if normalize:
                    ylabel = "Fraction"
                else:
                    ylabel = "Count"
            
                ax.set_ylabel(ylabel)
            else:
                print(attr + "\n")
                print(plot_data)
                print("\n")


            fig.tight_layout()

    def plot_cond(self, df, var, conds, ax, normalize=True):
        """
        Plot probability of a value in column var of DataFrame df, conditional on conditions expressed in conds

        Parameters
        ----------
        df: DataFrame
        var: String.  Name of column in df whose density we will plot
        conds: Dictionary
        - keys are Strings, which are names of columns in df
        - values are values that could be compared with column at the key
        
        normalize: Boolean. If True, display relative (vs absolute) values

        """
        plot_data = df.copy()
        title_array = []

        for cond, val in conds.items():
            title_array.append( "{c}={v}".format(c=cond, v=val))
            plot_data = plot_data.loc[ plot_data.loc[:, cond] == val, : ]

            args = { "kind": "bar"}


        plot_data = plot_data.loc[:, var ]

        title = ", \n".join(title_array)
        title = "Prob({v} | {t})".format(v=var, t=title)
        plot_data.value_counts(normalize=normalize).sort_index().plot(title=title, ax=ax, **args)

    def plot_conds(self, df, specs, share_y=None, normalize=True):
        """
        Print multiple conditional plots using plot_cond

        Parameters
        -----------
        df: DataFrame
        specs: List. Each element of the list is a tuple (var, conds)
        -  each element of the list generates a call to plot_cond(df, var, conds)

        share_y: Boolean.  Option "sharey" argument to matplotlib
        - share_y == "row": all plots on same row share y-axis
        """
        num_specs = len(specs)
        ncols=3
        nrows = max(1,round(.4999 + num_specs/ncols))

        args = {}
        if share_y is not None:
            args["sharey"] = share_y
            
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(ncols*4,
                                          min(8, num_specs*1.5)
                                          ),
                                 **args
                                 )

        # Make sure axes is an array (special case when num_attrs==1)
        if num_specs == 1:
            axes =np.array( [ axes ])

        for i, spec in enumerate(specs):
            if spec is None:
                continue
            (var, conds) = spec
            self.plot_cond(df, var, conds, ax=axes.flatten()[i], normalize=normalize)

        fig.tight_layout()

        return fig, axes

    def AUC_plot(self,  X_train=None, X_test=None, y_train=None, y_test=None):
        if X_train is None:
            X_train = self.X_train

        if X_test is None:
            X_test = self.X_test

        if y_train is None:
            y_train = self.y_train

        if y_test is None:
            y_test = self.y_test
            

        clf_lr = LogisticRegression(C=50. / X_train.shape[0],  # n.b. C is 1/(regularization penalty)
                                    multi_class='multinomial',
                                    # penalty='l1',   # n.b., "l1" loss: sparsity (number of non-zero) >> "l2" loss (dafault)
                                    solver='saga', tol=0.1)

        clf_rf =  RandomForestClassifier(n_estimators=10, random_state=42)

        clf_knn = KNeighborsClassifier(n_neighbors=3)

        # fit a model
        models = { "Logistic Regression": clf_lr,
                   "Random Forest": clf_rf,
                   "KNN": clf_knn
        }

        fig, axs = plt.subplots(1, len(models), figsize=(12,6))

        model_num = 0
        
        # Fit each model and plot an ROC curve
        for name, model in models.items():
            # Axis for this model
            ax = axs[model_num]
            model_num += 1

            if self.Debug:
                print(name)
                
            model.fit(X_train, y_train)

            # predict probabilities
            probs = model.predict_proba(X_test)

            # keep probabilities for the positive outcome only
            # n.b., roc_auc_score needs scores, not probabilities
            # - some models can do this via the "decision_funciton" arg.
            # - for those that can't, just use the probability of a single class
            scores = probs[:, 1]

            # calculate AUC
            auc = roc_auc_score(y_test, scores)

            # calculate roc curve
            fpr, tpr, thresholds = roc_curve(y_test, scores)
            # plot no skill
            ax.plot([0, 1], [0, 1], linestyle='--')
            # plot the roc curve for the model
            ax.plot(fpr, tpr, marker='.')


            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            
            ax.set_title( "{m:s} (AUC={auc:3.2f})".format(m=name, auc=auc) )

        fig.tight_layout()

class NB_Helper():
    def __init__(self, **params):
        return

    def gen_fin_cluster(self,
                        n_samples=100,
                        feature_names=[],
                        target_name="target",
                        class_names=[],
                        feature_to_value=None,
                        random_state=10,
                        n_informative=None, n_redundant=0, n_repeated=0,
                        n_clusters_per_class=1
                        ):

        n_features = len(feature_names)
        n_classes  = len(class_names)
        
        if n_informative is None:
            n_informative = n_features
                        

        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                  n_informative=2, 
                                  n_redundant=0,
                                  n_repeated=0,
                                  n_clusters_per_class=1,
                                  class_sep=1.0,
                                  random_state=random_state
                                  )

        # Map class integers to class_names
        y_classes = [ class_names[c_num] for c_num in y ]

        # DataFrames of numbers
        X_df = pd.DataFrame(X, columns=feature_names)
        y_df = pd.DataFrame(y, columns=[target_name])

        df = pd.concat( [X_df, y_df], axis=1)
        df_sorted = df.sort_values(by=feature_names)

        # Create DataFrames of labels
        X_label_df = X_df.copy()
        y_label_df = y_df.copy()

        y_label_df = pd.DataFrame(class_names, columns=["target"])

        
        # Map feature values to names
        if feature_to_value is not None:
            # For each feature: map values to names
            for feat, feat_list in feature_to_value.items():
                num_feats = len(feat_list)

                # Bucket the values in column "feat"
                X_label_df[ feat ] = pd.cut(X_label_df[feat], bins=len(feat_list),
                                                     labels=feat_list)

                
        return X_label_df, y_label_df, X_df,y_df,

    def map_num_to_cat(self, X, y, feat_map, class_map):
        X_new_dict = {}
        
        # Map each feature
        for feat_name, feat_list in feat_map.items():
            # Create a series (column values) for the new values of the feature
            labels = pd.Series( [ None ] * X.shape[0] )
            X_new_dict[feat_name] = labels
            
            # Apply each mapping for feature feat_name
            for i, one_map in enumerate(feat_list):
                min_val, max_val, label = [ one_map[k] for k in ("min", "max", "class") ]
                labels[ (X[feat_name] > min_val) & (X[feat_name] <= max_val) ] = label

        # Create data frame from X_new_dict
        X_new = pd.DataFrame.from_dict(X_new_dict)

        # Map the targets
        targets = pd.Series( [ None ] * y.shape[0] )
        target_name = y.columns[0]
        for class_num, label in class_map.items():
            targets[ y[target_name] == class_num ] = label

        y_new = pd.DataFrame.from_dict( { target_name: targets } )
        
        return X_new,  y_new
            
    def gen_stock_recommend_data(self):
        feature_names = ["Valuation", "Yield"]

        # n.b., can't control the order of class 
        class_names   = [ "Short", "Neutral", "Long" ]

        feature_to_value = { "Valuation": [ "Rich", "Fair", "Cheap"],
                             "Yield": [ "Low", "High"]
                             }


        X_df, y_df, X_num_df, y_num_df = self.gen_fin_cluster(n_samples=20, 
                                                             feature_names=feature_names, class_names=class_names,
                                                             feature_to_value=feature_to_value,
                                                             random_state=10
                                                             )
        


        class_map, feat_map = {},  {}
        class_map = { 0: "Neutral", 1: "Short", 2: "Long"}

        feat_map["Valuation"] = [ {"min": -10,  "max": -1, "class": "Rich" },
                                  {"min": -1,  "max":   0.6, "class": "Fair" },
                                  {"min":  0.6,  "max":  10, "class":  "Cheap" }
                                  ]

        feat_map["Yield"]   = [ { "min": -10, "max":-1.0 , "class": "Low" }, 
                                { "min": -1.0,    "max": 10, "class": "High" }
                                ]


        X_new, y_new = self.map_num_to_cat(X_num_df, y_num_df, feat_map, class_map)
        new_df = pd.concat([X_new, y_new], axis=1)
        new_df.index.name = "#"

        return new_df
    
class LinearSep_Helper():
    def __init__(self, **params):
        self.Debug = False

        self.clh = Classification_Helper()
        
        return

    def load_iris(self, binary=True, scale=True):
        iris = datasets.load_iris()
        X = iris["data"][:, (2, 3)]  # petal length, petal width
        y = iris["target"]

        if binary:
            setosa_or_versicolor = (y == 0) | (y == 1)
            X = X[setosa_or_versicolor]
            y = y[setosa_or_versicolor]

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X, y

    def plot_y_vs_x(self, ax, X, y):
        _= ax.scatter(X[ y <= 0], y[ y <= 0], c="red",   label="0")
        _= ax.scatter(X[ y > 0 ],  y[ y > 0],  c="green", label="1")

        _= ax.set_xlabel('$x$')
        _= ax.set_ylabel('$y$')
        _= ax.legend()

        return ax

    def plot_sigmoid(self, ax):
        s = np.linspace(-10,10, 50)
        sigma_s = 1/(1 + np.exp(- s))

        _= ax.plot(s, sigma_s)
        _= ax.set_title("Sigmoid")
        _= ax.set_xlabel("score = $\Theta^T x$")
        _= ax.set_ylabel("probability")

        return ax
    
    def fit_LR(self, X, y):
        clf = LogisticRegression(solver='lbfgs', C=100)
        clf.fit(X, y)

        return clf

    def fit_SVC(self, X, y):
        clf = SVC(kernel="linear", C=100)
        clf.fit(X, y)

        return clf

    def boundary(self, clf, x0):
        """
        Create the 2D line that is the boundary between binary labels

        Parameters
        ----------
        clf: an sklearn classifier
        x0: np.array.  Values of x_0 for which the correspond y is computed t the boundary
        """
        # Get the classifier's coefficients: assumption is that it is 2D
        w = clf.coef_[0]
        b = clf.intercept_[0]

        # At the decision boundary, w0*x0 + w1*x1 + b = 0
        # => x1 = -w0/w1 * x0 - b/w1
        xcept = b/w[1]
        slope = -w[0]/w[1]
        x1 = slope * x0 - xcept

        return xcept, slope, x1

    def score_to_prob(self, score):
        """
        Convert score to probability according to formula
        - score = ln(p/(1-p))

        Parameters
        ----------
        score: float

        Returns
        -------
        prob: Float, probability corresponding to score
        """
        prob = np.exp(score)/(1+np.exp(score))
        return prob

    def plot(self, ax, clf, X, y, draw_boundary=True, draw_prob=True, scores=[0]):
        """
        Plot the points and the decision boundary
        - optionally plot lines of constant score

        Parameters
        ----------
        ax: matplotlib Axes in which to plot
        clf: sklearn classifier (assuming 2D)
        X, y: the dataset
        - X is assumed to be vectors of length 2
        - y is assumed to be a binary label

        Optional
        --------
        scores: ndarray.  Scores for which constant lines will be drawn
        """
        # Create a 2D grid of (x_0, x_1) values
        # We will use this to plot the decision
        X0, X1 = np.meshgrid(np.arange(start = X[:, 0].min() - 1,
                                       stop = X[:, 0].max() + 1,
                                       step = 0.01),
                             np.arange(start = X[:, 1].min() - 1,
                                       stop = X[:, 1].max() + 1,
                                       step = 0.01)
        )


        if draw_boundary:
            clh = self.clh

            _= clh.plot_boundary_2(clf, X, y, ax=ax, cmap=ListedColormap(('navajowhite', 'darkkhaki')), scatter=False)


        _= ax.set_xlim(X0.min(), X0.max())
        _= ax.set_ylim(X1.min(), X1.max())

        # Scatter plot, color coded by class label (y)
        for i, j in enumerate(np.unique(y)):
            _= plt.scatter(X[y == j, 0], X[y == j, 1],
                           c = ('red', 'green')[i], label = j)

        _= ax.set_xlabel('$x_0$')
        _= ax.set_ylabel('$x_1$')
        _= ax.legend()

        # Create the boundary line, stretching from X0.min() to X0.max()
        x0 = np.linspace(X0.min(), X0.max(), 200)
        xcept, slope, decision_boundary = self.boundary(clf, x0)
        
        l2 = np.array([0,0])

        # Express slope of line as an angle, in degrees
        angle =  np.arctan(slope) * 180/np.pi

        # Translate the angle (slope)
        # See: https://matplotlib.org/3.1.1/tutorials/advanced/transforms_tutorial.html
        # - from the data coordinate system (ax.transData)
        # - to the coordinate system of the plot
        # --- i.e., the horizontal and vertical axes may not be same scale
        trans_angle = ax.transData.transform_angles(np.array((angle,)),
                                                           l2.reshape((1, 2))
                                                    )[0]

        # Create a line of constant score, for various constants
        for s in scores:
            pr =self.score_to_prob(s)
    
            ax.plot(x0, decision_boundary + s, "k-", linewidth=2)

            
            eqn = "score={s:3.2f}".format(s=s)

            if draw_prob:
                eqn = eqn + " (prob={pr:3.2f})".format(pr=pr)
                
            _= ax.text(0, s, eqn,
                       horizontalalignment='center',
                       verticalalignment='center',
                       rotation=trans_angle
                       )
            
        _= plt.show()

            

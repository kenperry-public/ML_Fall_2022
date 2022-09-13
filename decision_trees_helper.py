import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import pdb

import os
import subprocess

# sklearn
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.base import BaseEstimator, TransformerMixin

# Tools
from sklearn import preprocessing, model_selection 
from sklearn.tree import export_graphviz
from sklearn import tree

# Models
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

# Datasets
from sklearn import datasets

# Other classes
import class_helper

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

class SexToInt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        I am really cheating here ! Am ignoring all columns except for "Sex"
        """
        
        # To see that I am cheating, look at the number of columns of X !
        print("SexToInt:transform: Cheating alert!, X has {c} columns.".format(c=X.shape[-1]) )
        
        sex = X["Sex"]
        X["Sex"] = 0
        X[ sex == "female" ] = 1
        
        return(X)

class TitanicHelper():
    def __init__(self, **params):
        TITANIC_PATH = os.path.join("./data", "titanic")

        if not os.path.isdir(TITANIC_PATH):
            print("TitanicHelper: you MUST download the data to ", TITANIC_PATH)
            
        self.TITANIC_PATH = TITANIC_PATH

        train_data = pd.read_csv( os.path.join(TITANIC_PATH, "train.csv") )
        test_data  = pd.read_csv( os.path.join(TITANIC_PATH, "test.csv")  )

        target_name = "Survived"

        train_data = train_data[ train_data[target_name].notnull() ]

        y_train = train_data[target_name]
        X_train = train_data.drop(columns=["Survived"], inplace=False)
        
        self.train_data = train_data
        self.test_data  = test_data
        self.target_name = target_name
        self.X_train, self.y_train = X_train, y_train

        return

    def make_numeric_pipeline(self, num_features):
        num_transformers= Pipeline(steps=[  ('imputer', SimpleImputer(strategy='median')) ] )
        num_pipeline = ColumnTransformer( transformers=[ ("numeric", num_transformers, num_features) ] )
        
        return num_transformers

    def make_cat_pipeline(self, cat_features, drop=None):
        cat_transformers= Pipeline(steps=[  ('imputer', SimpleImputer(strategy="most_frequent")),
                                            ('cat_encoder', OneHotEncoder(sparse=False, categories="auto", drop=drop))
                                            # ('cat_encoder', OrdinalEncoder())
                                            ]
                                   )

        cat_pipeline = ColumnTransformer( transformers=[ ("categorical", cat_transformers, cat_features) ] )


        self.is_ohe = True
        self.drop_cat = drop
        
        return cat_transformers

    def make_pipeline(self, num_features=["Age", "SibSp", "Parch", "Fare" ],
                      cat_features= ["Sex", "Pclass" ],
                      drop_cat="first"
                      ):

        # Create numeric and categorical pipelines
        num_transformers = self.make_numeric_pipeline(num_features)
        cat_transformers = self.make_cat_pipeline(cat_features, drop=drop_cat)

        # Create combined pipeline
        preprocess_pipeline = ColumnTransformer(
            transformers=[ ("numeric", num_transformers, num_features),
                           ("categorical", cat_transformers, cat_features)
                           ]
            )

        # feature_names MUST BE in same order as the pipeline
        feature_names = num_features.copy()
        feature_names.extend(cat_features)

        # Record the feature names in the same order in which they were created by the pipeline
        self.feature_names = feature_names

        self.num_features = num_features
        self.cat_features = cat_features
        
        return preprocess_pipeline, feature_names

    def run_pipeline(self, pipeline, data):
        # Run the pipelinem return an ndarray
        data_trans = pipeline.fit_transform(data)

        return data_trans


    def make_logit_clf(self):
        # New version of sklearn will give a warning if you don't specify a solver (b/c the default solver -- liblinear -- will be replaced in future)
        logistic_clf = linear_model.LogisticRegression(solver='liblinear')

        return logistic_clf

    def make_tree_clf(self, max_depth):
        tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        return tree_clf

    def fit(self, clf, drop_cat="first"):
        X_train, y_train = self.X_train, self.y_train

        # Create the Transformation pipeline
        preprocess_pipeline, feature_names = self.make_pipeline(drop_cat=drop_cat)

        # Combine Transformations with Classifier into one pipeline
        model_pipeline = Pipeline(steps=[ ("transform", preprocess_pipeline),
                                          ("classify", clf)
                                    ]
                             )

        # Cross validation
        scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
        self.scores = scores

        # Final Fit on the entire training data
        model_pipeline.fit(X_train, y_train)
        
        return model_pipeline

    def export_tree(self, tree_clf, out_file,
                    feature_names, target_classes,
                    to_png=True,
                    use_graphviz=False,
                    **params
                    ):

        # Returns a dict
        ret = {}
        
        if use_graphviz:
            # Use graphviz to create a ".dot" file, which will be converted to a png
            dot_file = out_file + ".dot"

            ret = { "dot_file": dot_file }

            export_graphviz(
                tree_clf,
                out_file=dot_file,
                feature_names=feature_names,
                class_names=target_classes,
                rounded=True,
                filled=True
                )

            if to_png:
                # Convert .dot file to a png
                # NOTE: uses an external process running
                #  command "dot", which needs to be installed
                png_file = out_file + ".png"
                cmd = "dot -Tpng {dotf} -o {pngf}".format(dotf=dot_file, pngf=png_file)
                ret["png_file"] = png_file

                retval = subprocess.call(cmd, shell=True)
                ret["dot cmd rc"] = retval


        # Use plot_tree to create a tree in matplotlib
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        _= tree.plot_tree(tree_clf,
                          feature_names=feature_names,
                          class_names=target_classes,
                          rounded=True,
                          filled=True, 
                          label="all",
                          fontsize=14, ax=ax,
                          **params
                          )
        
        ret["plt"] = { "fig": fig, "ax": ax }
        
        return ret

    def make_png(self, clf, out_file, feature_names, target_classes,
                 max_depth=2):
        ret = self.export_tree(clf, out_file, feature_names, target_classes, to_png=True )

        return ret

    def titanic_feature_names(self, X_train, num_features=[], cat_features=[],
                              is_ohe=True, drop_cat=None):
        """
        After One Hot Encoding (OHE) categorical features:
        - a feature with the original name no longer exists
        - it has been replaced by a number of binary indicators
        - so replace the original name in list of features with names for the indicators

        Parameters
        ----------
        X_train: DatFrame or ndarray.  Training data
        - needed if categorical features were OHE, in order to have access to the values of the categories in a categorical features

        num_features: List of strings.  Names of numeric features
        cat_features: List of strings.  List of categorical features

        is_ohe: Boolean.  Indicates whether the (originally) categorical features have been replace by OHE indicators
        drop_cat: String.  If not None: is the argument passed to OHE to indicate which feature to drop, e.g., "first", if any.
        """
        
        if is_ohe:
            drop_cat = self.drop_cat
            first_idx = 0 if drop_cat is None else 1

            # The categorical features have been One Hot Encoded
            # - must assign a name ("string") to each category, based on the OHE
            X_train = self.X_train
            feature_names = num_features.copy()

            cat_features = cat_features

            # One Hot Encode (OHE) the categorical features so that we have access to the names of the cateories in each
            ohe = OneHotEncoder(sparse="False", categories="auto", drop=drop_cat).fit(X_train[ cat_features] )
            for i, feat in enumerate(cat_features):
                # Extract the names of the categories for categorical feature i
                categories = [ "{f:s} is {c:s}".format(f=feat, c=str(c)) for c in ohe.categories_[i][first_idx:] ]

                # Add new binary indicator feature with name "Is c" for each category in feature i
                feature_names.extend(categories)

        return feature_names
                
    def make_titanic_png(self, drop_cat=None, max_depth=2, **params):
        train_data = self.train_data
        target_name = self.target_name

        tree_clf = self.make_tree_clf(max_depth=max_depth)
        pipeline = self.fit(tree_clf, drop_cat=drop_cat)

        fname = "images/titanic_{depth:d}level".format(depth=max_depth)

        # Replace name of each categorical feature with a list of names for each indicator (category) if categrical variables have been One Hot Encoded
        feature_names = self.titanic_feature_names(self.X_train,
                                                   num_features=self.num_features,
                                                   cat_features=self.cat_features,
                                                   is_ohe=self.is_ohe,
                                                   drop_cat=drop_cat
                                                   )


        ret = self.export_tree(tree_clf, fname,
                               feature_names, #self.feature_names,
                               [ "No", "Yes"],
                               **params
                               )

        ret["fname"] = fname
        ret["pipeline"] = pipeline

        return ret


    def partition(self, X, y, conds=[]):
        mask = pd.Series(data= [ True ] * X.shape[0], index=X.index )
        X_filt = X.copy()

        for cond in conds:
            (col, thresh) = cond
            print("Filtering column {c} on {t}".format(c=col, t=thresh) )
            cmp = X[ col ] <= thresh
            mask = mask & cmp

        return (X[mask], y[mask], X[~mask], y[~mask])
            

class GiniHelper():
    def __init(self, **params):
        return

    def plot_Gini(self):
        """
        Plot Gini score of binary class
        """
        p     = np.linspace(0,1, 1000)
        not_p = 1 - p
        
        gini = 1 - np.sum(np.c_[p, not_p]**2, axis=1)
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        _ = ax.plot(p, gini)
        _ = ax.set_xlabel("p")
        _ = ax.set_ylabel("Gini")
        _ = ax.set_title("Gini: Two class example")

        return fig, ax

    def make_logic_fn(self, num_features=3, target_name="target"):
        rows = []
        fstring = "{{:0{:d}b}}".format(num_features)

        for i in range(0, 2**num_features):
            row =[ int(x) for x in list( fstring.format(i) ) ]
            rows.append(row)

        feature_names = [ "feat {i:d}".format(i=i)  for i in range(1,num_features+1) ]
        df = pd.DataFrame.from_records(rows, columns=feature_names)

        target =  ( (df["feat 1"] == 1) | df["feat 2"] == 1 ) & (df["feat 3"] == 1)

        df[target_name] = target.astype(int)

        return df, target_name, feature_names
        
    def make_logic_dtree(self, df, target_name):
        tree_clf = DecisionTreeClassifier( random_state=42 )
        
        y = df[target_name]
        X = df.drop(columns=[target_name])
        tree_clf.fit(X, y)

        return tree_clf

    def make_logicTree_png(self):
       df, target_name, feature_names = self.make_logic_fn()

       self.df_lt, self.target_name_lt, self.feature_names_lt = df, target_name, feature_names

       th = TitanicHelper()
       tree_clf = self.make_logic_dtree(df, target_name)

       fname = "images/logic_tree"
       th.export_tree(tree_clf, fname, feature_names, [ "No", "Yes"] )

       return fname

    def gini(self, df, target_name, feature_names, noisy=False):
        
        count_by_target = df[ target_name ].value_counts()
        count_total = count_by_target.values.sum()

        # Compute frequencies
        freq = count_by_target/count_total

        # Square the frequencies
        freq2 = freq ** 2
        
        # Compute Gini
        gini = 1 - freq2.sum()

        if noisy:
            print("Gini, by hand:")
            print("Count by target:\n\t")
            print(count_by_target)
            print("Frequency by target:\n\t")
            print(freq)
            print ("\n1 - sum(freq**2) = {g:0.3f}".format(g =gini) )

        return gini

    def cost(self, df, target_name, feature_names, noisy=False):
        for feature_name in feature_names:
            count_by_feature_value = df[feature_name].value_counts()
            count_total = count_by_feature_value.values.sum()

            feature_values = count_by_feature_value.index.tolist()
            
            # Eliminate the max value since <= max includes everything
            feature_values = sorted(feature_values)[:-1]

            for feature_value in feature_values:
                cond = df[feature_name] <= feature_value
                df_left = df[ cond ]
                df_right = df[ ~ cond ]

                gini_left  = self.gini(df_left, target_name, feature_names)
                gini_right = self.gini(df_right, target_name, feature_names)
                
                count_left  = df_left.shape[0]
                count_right = df_right.shape[0]
                

                cost = (count_left/count_total) * gini_left + (count_right/count_total) * gini_right

                if noisy:
                    print("Split feature {f:s} on {fv:0.2f}".format(f=feature_name, fv=feature_value))
                    print("\tG_left (# = {lc:d}) = {gl:0.3f}, G_right (# = {rc:d}) = {gr:0.3f}".format(gl=gini_left, gr=gini_right, lc=count_left, rc=count_right) )
                    print("\tweighted (G_left, G_right) = {c:0.3f}".format(c=cost) )
        return cost

class RegressionHelper():
    def __init__(self, **params):
        return

    def make_plot(self, seed=42):
        """
        Based on https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
        """
        # Create a random dataset
        rng = np.random.RandomState(seed)

        X = np.sort(5 * rng.rand(80, 1), axis=0)
        y = np.sin(X).ravel()
        y[::5] += 3 * (0.5 - rng.rand(16))

        # Predict
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

        th = TitanicHelper()

        # Fit and  Plot the results
        fig, ax = plt.subplots(1,2, figsize=(12,5))

        fig_trees, ax_trees = [], []
        
        for i, depth in enumerate([2,5]):
            regr = DecisionTreeRegressor(max_depth=depth)
            regr.fit(X, y)

            y_1 = regr.predict(X_test)

            ax[i].scatter(X, y, s=20, edgecolor="black",
                          c="darkorange", label="data")

            ax[i].plot(X_test, y_1, color="cornflowerblue",
                 label="max_depth={d:d}".format(d=depth), linewidth=2)

            ax[i].set_xlabel("data")
            ax[i].set_ylabel("target")
            ax[i].set_title("Decision Tree Regression, max depth={d:d}".format(d=depth) )

            # Create the png
            fname = "images/tree_regress_depth_{d:d}".format(d=depth)
            ret_tree = th.export_tree(regr, fname, [ "X" ], [ "No", "Yes"] )
            fig_tree, ax_tree = ret_tree["plt"]["fig"], ret_tree["plt"]["ax"]
            plt.close(fig_tree)

            # Collect the plots>
            fig_trees.append(fig_tree)
            ax_trees.append(ax_tree)

        plt.close(fig)
        
        return { "fig1":fig, "ax1": ax,
                 "fig2":fig_trees, "ax2": ax_trees
                 }

class Boundary_Helper:
    def __init__(self, **params):
        # Helper from classification task
        self.clh = class_helper.Classification_Helper()
        
        return

    
    def make_iris_2class(self):
        iris = datasets.load_iris()
        X = iris["data"][:, (2, 3)]  # petal length, petal width
        y = iris["target"]

        # Turn 3 classes into 2
        mask = (y == 0) | (y == 2)
        y[  mask ] = 0
        y[ ~mask ] = 1

        return X, y
    

    def make_boundary(self, X, y, depth=3, ax=None):
        clh = self.clh

        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))

        tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        _= tree_clf.fit(X, y)
        _= clh.plot_boundary_2(tree_clf, X, y, ax=ax)

        self.clf = tree_clf

        return ax


class Ensemble_Helper():
    def __init__(self, **params):
        # Helper from classification task
        self.clh = class_helper.Classification_Helper()
        
        return

    def make_iris(self, models=None):
        # Load data
        iris = datasets.load_iris()

        self.data = iris

        # Parameters
        n_classes = 3
        n_estimators = 30

        if models is None:
            models = [ RandomForestClassifier(max_depth=2,
                                              n_estimators=n_estimators
                                              )
                       ]

        self.models = models
        
        return iris

    def make_ens(self,
                 pairs=([0, 1], [0, 2], [2, 3]),
                 max_show=3
                 ):
        """
        Derived from: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#sphx-glr-auto-examples-ensemble-plot-forest-iris-py

        Run an ensemble classifer on data set
        - 2 features at a time
        -- from a list of feature pairs


        Parameters
        ----------
        pairs: array.  array of pairs. The pair are 2 indices of 2 features
        max_show: Int.  Maximum number of individual models in the ensemble to show

        Returns
        -------
        There is
        - one result per pair, per model, so arrays are doubly indexed

        estimators: array
        - estimators[p][m]: list of individual models for pair p, model m

        model_data: array
        - model_data[p][m] = tuple (X,y), the training data used to train model m for pair p

        """
        clh = self.clh
        data = self.data
        models = self.models

        cmap = plt.cm.RdYlBu

        plot_step = 0.02  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        RANDOM_SEED = 13  # fix the seed on each iteration

        # Create matplotlib figures
        # - for individual models in the ensemble
        # - for the ensemble itself
        len_fig = min(12, len(pairs)*6)
        fig, axs =         plt.subplots( len(pairs), max_show, figsize=(12,len_fig))
        fig_sum, axs_sum = plt.subplots( len(pairs), 2,        figsize=(12,len_fig))

        axs = axs.reshape( len(pairs), axs.shape[-1] )
        axs_sum = axs_sum.reshape( len(pairs), axs_sum.shape[-1] )
        
        estimators = []
        model_data = []
        
        # Create model, show individual models, show ensemble
        # - for each pair of features in pairs
        for p, pair in enumerate( pairs ):
            # List of estimators for this pair
            est_pair = []
            data_pair = []

            model_data.append(data_pair)
            estimators.append(est_pair)

            # Names of features for this pair
            feature_names=[ "$x_{i:d}$".format(i=idx+1)                                                              for i, idx in enumerate(pair)
                            ]

            # Do it for model in the list
            for m, model in enumerate(models):
                # We only take the two selected features in pair
                X = data.data[:, pair]
                y = data.target

                # Shuffle
                idx = np.arange(X.shape[0])
                np.random.seed(RANDOM_SEED)
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]

                # Standardize
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                X = (X - mean) / std

                data_pair.append( (X,y)  )

                # Train
                model.fit(X, y)

                scores = model.score(X, y)
                # Create a title for each column and the console by using str() and
                # slicing away useless parts of the string
                model_title = str(type(model)).split(
                    ".")[-1][:-2][:-len("Classifier")]

                model_details = model_title
                if hasattr(model, "estimators_"):
                    model_details += " with {} estimators".format(
                        len(model.estimators_))
                print(model_details + " with features", pair,
                      "has a score of", scores)


                # Plot either a single DecisionTreeClassifier or alpha blend the
                # decision surfaces of the ensemble of classifiers
                if isinstance(model, DecisionTreeClassifier):
                    # Now plot the decision boundary using a fine mesh as input to a
                    # filled contour plot
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                         np.arange(y_min, y_max, plot_step))


                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    if orig:
                        cs = plt.contourf(xx, yy, Z, cmap=cmap)
                else:
                    # Choose alpha blend level with respect to the number
                    # of estimators
                    # that are in use (noting that AdaBoost can use fewer estimators
                    # than its maximum if it achieves a good enough fit early on)

                    ax_blended = axs_sum[p, 0]
                    ax_ens     = axs_sum[p, 1]

                    # Blend the individual models together
                    # - compute the blending "alpha" to use in plot
                    alpha_div = min(max_show, len(model.estimators_))
                    estimator_alpha = 1.0 /alpha_div

                    # Enumerate the individual sub-models of the ensemble

                    this_p_m = []
                    est_pair.append(this_p_m)
                    
                    for e,tree in enumerate(model.estimators_):
                        if e >= max_show:
                            break
                        
                        this_p_m.append(tree)
                        
                        ax =  axs[p,e]
                        _= clh.plot_boundary_2(tree, X, y, ax=ax,
                                               feature_names=feature_names,
                                               alpha=estimator_alpha,
                                               show_legend=False)

                        # Contour for plot of BLEND of individual sub-model
                        # - alpha blended since same plot gets updated with each sub-model
                        _= clh.plot_boundary_2(tree, X, y, ax=ax_blended, alpha=estimator_alpha, show_legend=False)

                        ax.set_title("Sub-model {s:d}".format(s=e))
                        
                # Blended: plot training
                ax_blended.scatter(X[:, 0], X[:, 1], c=y,
                                   cmap=ListedColormap(['r', 'y', 'b']),
                                   edgecolor='k', s=20)
                ax_blended.set_title("Blended")

                # Ensemble: plot contour
                _= clh.plot_boundary_2(model, X, y, ax=ax_ens)
                
                ax_ens.set_title("Ensemble")


        fig.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
        fig.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
        plt.close(fig)

        fig_sum.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
        fig_sum.tight_layout()
        plt.close(fig_sum)

        result = { "individual models plot": [ fig, axs ],
                   "ensemble plot": [ fig_sum, axs_sum ],
                   "estimators": estimators,
                   "individual model data": model_data,
                   "pairs": pairs
            }
        
        return result

    def plot_ens(self, X, y, ens_estimators, feature_names=None):

        if feature_names is None:
            feature_names = [ "$x_{idx:d}$".format(idx=i+1) for i in range(0, X.shape[-1]) ]
        fig_est, axs_est = plt.subplots(1, len(ens_estimators), figsize=(15,9))
        axs_est = np.ravel(axs_est)
        for e, est in enumerate(ens_estimators):
            ax = axs_est[e]
            est = ens_estimators[e]
            _= tree.plot_tree(est,
                              feature_names=feature_names,
                              rounded=True,
                              filled=True,
                              label="all",
                              fontsize=8,
                              ax=ax
            )
    
     
        plt.close(fig_est)
        return fig_est

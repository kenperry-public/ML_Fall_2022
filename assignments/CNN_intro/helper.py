import math

import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import pickle
import tensorflow as tf
from nose.tools import assert_equal
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


import json

import pdb

class Helper():
  def __init__(self):
    # Data directory
    self.DATA_DIR = "./Data"

    if not os.path.isdir(self.DATA_DIR):
        self.DATA_DIR = "../resource/asnlib/publicdata/ships_in_satellite_images/data"

    self.dataset =  "shipsnet.json"

  def getData(self):
    
    data,labels = self.json_to_numpy( os.path.join(self.DATA_DIR, self.dataset) )
    return data, labels

  def showData(self, data, labels, num_cols=5, cmap=None):
    # Plot the first num_rows * num_cols images in X
    (num_rows, num_cols) = ( math.ceil(data.shape[0]/num_cols), num_cols)

    fig = plt.figure(figsize=(10,10))
    # Plot each image
    for i in range(0, data.shape[0]):
        img, img_label = data[i], labels[i]
        ax  = fig.add_subplot(num_rows, num_cols, i+1)
        _ = ax.set_axis_off()
        _ = ax.set_title(img_label)

        _ = plt.imshow(img, cmap=cmap)
    fig.tight_layout()

    return fig

  def modelPath(self, modelName):
      return os.path.join(".", "models", modelName)

  def y_OHE(self, y):
    """
    Determine the encoding of y
    - False if it is one dimensional (or two dimensional with final dimension of 1
    - True if it is One Hot Encoded

    Parameters
    -----------
    y: ndarray

    Returns
    -------
    Bool: 
    - True if y is OHE
    - False otherwise
    """
    result = None
    if ( (y.ndim > 1) and (y.shape[-1] >1) ):
      result = True
    else:
      result = False

    return result


  def saveModel(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      # Save model JSON to disk
      json_config = model.to_json()
      with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
          json_file.write(json_config)

      # Save weights to disk
      model.save_weights(os.path.join(model_path, 'weights.h5'))

      # Save training config
      metrics = model.metrics_names
      loss    = model.loss
      if 'loss' in metrics:
        metrics.remove('loss')

      training_parms = { "metrics": metrics,
                         "loss"   : loss
                         }
      
      with open(os.path.join(model_path, 'training_parms.pkl'), 'wb') as f:
          pickle.dump(training_parms, f)

      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))

  def loadModel(self, modelName):
    model_path = self.modelPath(modelName)

    # Reload the model from the files we saved
    with open(os.path.join(model_path, 'config.json')) as json_file:
        json_config = json_file.read()

    model = tf.keras.models.model_from_json(json_config)

    # Retrieve training parameters and restore them
    with open(os.path.join(model_path, 'training_parms.pkl'), 'rb') as f:
        training_parms = pickle.load(f)
        metrics, loss = ( training_parms[k] for k in ("metrics", "loss") )

    model.compile(loss=loss, metrics=metrics)
    model.load_weights(os.path.join(model_path, 'weights.h5'))

    return model

  def saveModelNonPortable(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      model.save( model_path )
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))
   
  def loadModelNonPortable(self, modelName):
      model_path = self.modelPath(modelName)
      model = self.load_model( model_path )
      
      # Reload the model 
      return model

  def saveHistory(self, history, model_name):
      history_path = self.modelPath(model_name)

      try:
          os.makedirs(history_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=history_path))

      # Save history
      with open(os.path.join(history_path, 'history'), 'wb') as f:
          pickle.dump(history.history, f)

  def loadHistory(self, model_name):
      history_path = self.modelPath(model_name)
      
      # Reload the model from the 2 files we saved
      with open(os.path.join(history_path, 'history'), 'rb') as f:
          history = pickle.load(f)
      
      return history

  def MyModel(self, test_dir, model_path):
      # YOU MAY NOT change model after this statement !
      model = self.loadModel(model_path)
      
      # It should run model to create an array of predictions; we initialize it to the empty array for convenience
      predictions = []
      
      # We need to match your array of predictions with the examples you are predicting
      # The array below (ids) should have a one-to-one correspondence and identify the example your are predicting
      # For Bankruptcy: the Id column
      # For Stock prediction: the date on which you are making a prediction
      ids = []
      
      # YOUR CODE GOES HERE
      
      
      return predictions, ids

  def json_to_numpy(self, json_file):
    # Read the JSON file
    f = open(json_file)
    dataset = json.load(f)
    f.close()

    data = np.array(dataset['data']).astype('uint8')
    labels = np.array(dataset['labels']).astype('uint8')

    # Reshape the data
    data = data.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

    return data, labels




  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
  modelName = "Ships_in_satellite_images"
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.01, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

  callbacks = [ es_callback,
                ModelCheckpoint(filepath=modelName + ".ckpt", monitor='accuracy', save_best_only=True)
                ]   

  max_epochs = 30

  def train(self, model, X, y, model_name, epochs=max_epochs):
    # Describe the model
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # Fix the validation set (for repeatability, not a great idea, in general)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print("Train set size: ", X_train.shape[0], ", Validation set size: ", X_valid.shape[0])

    history = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_valid, y_valid), callbacks=callbacks)
    fig, axs = plotTrain(history, model_name)

    return history, fig, axs

  def acc_key(self, history=None, model=None):
    """
    Parameters
    ----------
    model:   A Keras model object
    history: "history" object return by "fit" method applied to a Keras model

    Returns
    -------
    key_name: String.  The key to use in indexing into the dict contained in the history object returned by the "fit" method applied to a Keras model

    You should supply only ONE of these parameters (priority given to "model")

    Newer versions of Keras have changed the name of the metric that measures
    accuracy from "acc" to "accuracy".  Either name is allowed in the "compile" statement.

    The key in the history.history dictionary (returned by applying the "fit" method to the model object) will depend on the exact name of the metric supplied in the "compile" statement.

    This method will return the string to use as a key in history.history by examining
    - The model object (if given)
    - The keys of history.history (if history is given)
    """
    
    key_name = None
    
    if model is not None:
      key_name = "accuracy" if "accuracy" in model.metrics_names else "acc"
    else:
      key_name = "accuracy" if "accuracy" in history.history.keys() else "acc"

    return key_name

  def plotTrain(self, history, model_name="???"):
    fig, axs = plt.subplots( 1, 2, figsize=(12, 5) )

    # Determine the name of the key that indexes into the accuracy metric
    acc_string = self.acc_key(history=history)
    
    # Plot loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title(model_name + " " + 'model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
   
    # Plot accuracy
    axs[1].plot(history.history[ acc_string ])
    axs[1].plot(history.history['val_' + acc_string ])
    axs[1].set_title(model_name + " " +'model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')

    return fig, axs
  
  def model_interpretation(self, clf):
    dim = round( clf.coef_[0].shape[-1] **0.5)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1)
    scale = np.abs(clf.coef_[0]).max()
    _= ax.imshow( clf.coef_[0].reshape(dim, dim), interpolation='nearest',
                   cmap="gray",# plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)


    _ = ax.set_xticks(())
    _ = ax.set_yticks(())

    _= fig.suptitle('Parameters')

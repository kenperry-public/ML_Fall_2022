import math

import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import pickle

from nose.tools import assert_equal


import json

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

  def saveModel(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      # Save JSON config to disk
      json_config = model.to_json()
      with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
          json_file.write(json_config)
      # Save weights to disk
      model.save_weights(os.path.join(model_path, 'weights.h5'))
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))


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

      # Save JSON config to disk
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

    

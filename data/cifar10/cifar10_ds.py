# ......................................................................................
# MIT License

# Copyright (c) 2021-2024 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................

import pickle
import numpy as np
import os
from mllib import FileStore
from mllib.data import CDataSetBase
from data.cifar10.cifar10_downloader import DatasetDownloaderCIFAR10


# =========================================================================================================================
class CIFAR10Dataset(CDataSetBase):
  # --------------------------------------------------------------------------------------
  def __init__(self, datasets_folder, variation="", is_verbose=False):
    super(CIFAR10Dataset, self).__init__(p_sName="cifar10")  #means: ku-zu-shi-ji Modified National Institute of Standards and Technology
    # ................................................................
    self.variation = variation
    # // Fields \\
    self.fs = FileStore(datasets_folder)
    self.fs = self.fs.subfs(self.Name + self.variation)

    self.is_verbose = is_verbose


    self.ClassCount         = 10
    self.class_names        = [  "airplane", "automobile", "bird", "cat","deer"
                               , "dog", "frog", "horse", "ship", "truck"
                               ]
    self.feature_count      = 32*32*3
    self.images_shape        = [32, 32, 3]

    self.batches_info_file            = os.path.join(self.fs.base_folder, 'batches.meta')
    self.training_shard_filetemplate  = os.path.join(self.fs.base_folder, 'data_batch_%d')
    self.test_set_file                = os.path.join(self.fs.base_folder, 'test_batch')
    # ................................................................
  # --------------------------------------------------------------------------------------
  def load(self):
    if not self.LoadCache(self.fs, p_sTargetsFilePrefix="Labels"):
      self.download_and_prepare()
      self.SaveCache(self.fs, p_sTargetsFilePrefix="Labels")

    '''   
      self.fs.obj.save(self.TSSamples, "CIFAR10-TSSamples.pkl")
      self.fs.obj.save(self.TSTargets, "CIFAR10-TSLabels.pkl")
      self.fs.obj.save(self.VSSamples, "CIFAR10-VSSamples.pkl")
      self.fs.obj.save(self.VSTargets, "CIFAR10-VSLabels.pkl")
    else:
      self.TSSampleCount = self.TSSamples.shape[0]
      self.VSSampleCount = self.VSSamples.shape[0]
      self.SampleCount = self.TSSampleCount + self.VSSampleCount
      self.FeatureCount = np.prod(self.TSSamples.shape[1:])
      self.ClassCount = len(np.unique(self.TSTargets))
    '''
  # --------------------------------------------------------------------------------------
  def download_and_prepare(self):
    oDownloader = DatasetDownloaderCIFAR10(self.fs.base_folder)
    oDownloader.download()

    self.load_subset(True)
    self.load_subset(False)
    
    self.SampleCount      = self.TSSampleCount + self.VSSampleCount
    self.FeatureCount     = np.prod(self.TSSamples.shape[1:])
    self.ClassCount       = len(np.unique(self.TSTargets))

    print("Classes:", self.ClassCount)
  # --------------------------------------------------------------------------------------------------------
  def append_training_shard(self, samples, labels):
    # First shard initializes training set, next shards are appended
    if self.TSSamples is None:
      self.TSSamples = samples
      self.TSSampleCount = 0
    else:
      self.TSSamples = np.concatenate((self.TSSamples, samples), axis=0)

    if self.TSTargets is None:
      self.TSTargets = labels
    else:
      self.TSTargets = np.concatenate((self.TSTargets, labels), axis=0)
      
    self.TSSampleCount += samples.shape[0]
  # --------------------------------------------------------------------------------------------------------
  def append_validation_shard(self, samples, labels):
    # First shard initializes test (validation) set, next shards are appended
    if self.VSSamples is None:
      self.VSSamples = samples
      self.VSSampleCount = 0
    else:
      self.VSSamples = np.concatenate((self.VSSamples, samples), axis=0)

    if self.VSTargets is None:
      self.VSTargets = labels
    else:
      self.VSTargets = np.concatenate((self.VSTargets, labels), axis=0)

    self.VSSampleCount += samples.shape[0]
  # --------------------------------------------------------------------------------------------------------
  def _transposeImageChannels(self, image, shape=(32, 32, 3), is_flattening=False):
    """
    This method create image tensors (Spatial_dim, Spatial_dim, Channels) from image vectors of 32x32x3 features
    """
    nResult = np.asarray(image, dtype=np.uint8)#, dtype=np.float32)
    nResult = nResult.reshape([-1, shape[2], shape[0], shape[1]])
    nResult = nResult.transpose([0, 2, 3, 1])
        
    if is_flattening:
      nResult = nResult.reshape(-1, np.prod(np.asarray(shape)))
        
    return nResult 
  # --------------------------------------------------------------------------------------------------------
  def load_subset(self, is_training_set=True):
    if is_training_set:
      for i in range(5):
        with open(self.training_shard_filetemplate % (i + 1), 'rb') as oFile:
          oDict = pickle.load(oFile, encoding='latin1')
          oFile.close()
        self.append_training_shard(self._transposeImageChannels(oDict["data"], (32, 32, 3)), np.array(oDict['labels'], np.uint8))
    else:
      with open(self.test_set_file, 'rb') as oFile:
        oDict = pickle.load(oFile, encoding='latin1')
        oFile.close()
      self.append_validation_shard(self._transposeImageChannels(oDict["data"], (32, 32, 3)), np.array(oDict['labels'], np.uint8))
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
  import mllib as ml
  ml.RandomSeed(2024)
  IS_TESTING_NORMALIZATION  = True
  IS_TESTING_FEED           = True
  import matplotlib.pyplot as plt
  from data.cifar10.cifar10_feed import CCIFAR10DataFeed

  oDataSet = CIFAR10Dataset("C:\\MLData")
  oDataSet.load()
  oDataSet.PrintInfo()

  oFeed = None
  if IS_TESTING_NORMALIZATION:
    # Look at some sample images from dataset
    oDataSet.preview_images()

    if oFeed is None:
      config = {"InputShape": [32,32,3], "ClassCount":10, "Training.BatchSize": 128,
                "Validation.BatchSize": 128, "Prediction.BatchSize": 200}
      oFeed = CCIFAR10DataFeed(oDataSet, config)
      print(f"mean: {oFeed.PixelMean} std:{oFeed.PixelStd}")


    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      nImage = oDataSet.TSSamples[i].squeeze()
      nImageNormed = oFeed.normalizeImage(nImage)
      # nImageNormed = (nImage - oFeed.PixelMean)/oFeed.PixelStd
      plt.imshow(nImageNormed)
      plt.title(f"Label: {oDataSet.TSLabels[i]}")
      plt.axis('off')
    plt.show()


  if IS_TESTING_FEED:
    if oFeed is None:
      config = {"InputShape": [32,32,3], "ClassCount":10, "Training.BatchSize": 128,
                "Validation.BatchSize": 128, "Prediction.BatchSize": 200}
      oFeed = CCIFAR10DataFeed(oDataSet, config)
      print(f"mean: {oFeed.PixelMean} std:{oFeed.PixelStd}")

    (nSamples, nLabels) =  next(iter(oFeed.TSFeed))
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(nSamples[i].numpy().squeeze())
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(oFeed.denormalize_image(nSamples[i].numpy().squeeze()))
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()



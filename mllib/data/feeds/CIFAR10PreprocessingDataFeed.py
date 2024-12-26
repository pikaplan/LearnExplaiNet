# ......................................................................................
# MIT License

# Copyright (c) 2020-2024 Pantelis I. Kaplanoglou

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


import numpy as np
import tensorflow as tf


# =========================================================================================================================
class CCIFAR10PreprocessingDataFeed(object):
    
  # --------------------------------------------------------------------------------------
  def __init__(self, p_oDataSet, p_nBatchSize=None, p_nClassCount=None, p_bIsValidationOnly=False):
    super(CCIFAR10PreprocessingDataFeed, self).__init__()
    # ................................................................
    # // Fields \\
    self.DataSet      = p_oDataSet
    self.FileStore    = p_oDataSet.FileStore  
                  
    oDataSetStats = self.FileStore.Deserialize("%s-meanstd.pkl" % self.DataSet.Name)
    if oDataSetStats is not None:
      self.PixelMean    = oDataSetStats["mean"]
      self.PixelStd     = oDataSetStats["std"]
    else:
      self.calculateAndSaveDatasetStats()
    
    self.BatchSize  = p_nBatchSize
    self.ClassCount = p_nClassCount
    if p_nClassCount is None:
      self.ClassCount = self.DataSet.ClassCount
      
      
    
    if p_bIsValidationOnly:
      self.TSFeed = None
      self.VSFeed = self.CreateValidationDataFeed(self.BatchSize)
    else:
      self.TSFeed = self.CreateTrainingDataFeed()
      self.VSFeed = self.CreateValidationDataFeed()
    # ................................................................
  # --------------------------------------------------------------------------------------
  def calculateAndSaveDatasetStats(self):
    self.PixelMean = np.mean(self.DataSet.TSSamples, axis=(0,1,2))
    self.PixelStd  = np.std(self.DataSet.TSSamples, axis=(0,1,2))
    oDataSetStats = { "mean": self.PixelMean, "std": self.PixelStd}
    self.FileStore.Serialize("%s-meanstd.pkl" % self.DataSet.Name, oDataSetStats)
  # --------------------------------------------------------------------------------------------------------
  def normalizeImage(self, p_nImage):
      normed_image = (p_nImage - self.PixelMean) / self.PixelStd
      return normed_image     
  # --------------------------------------------------------------------------------------------------------
  def randomCropAndFlip(self, image):
      # We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side, 
      # and a  32×32  crop is  randomly  sampled  from  the  paddedimage or its horizontal flip.  
    
      #distorted_image = image
      distorted_image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)    # pad 4 pixels to each side
      distorted_image = tf.image.random_crop(distorted_image, [32, 32, 3])
      distorted_image = tf.image.random_flip_left_right(distorted_image)
      return distorted_image    
  # -----------------------------------------------------------------------------------
  def PreprocessTrainingImageAugmentDataset(self, p_tImageInTS, p_tLabelInTS):
    tNormalizedImage = self.normalizeImage(p_tImageInTS)
    tNewRandomImage  = self.randomCropAndFlip(tNormalizedImage)
    
    tTargetOneHot = tf.one_hot(p_tLabelInTS, self.ClassCount)
    
    return tNewRandomImage, tTargetOneHot
  # --------------------------------------------------------------------------------------
  def CreateTrainingDataFeed(self):
    oTSData = tf.data.Dataset.from_tensor_slices((self.DataSet.TSSamples, self.DataSet.TSLabels))
    oTSData = oTSData.map(self.PreprocessTrainingImageAugmentDataset, num_parallel_calls=8)
    oTSData = oTSData.cache()
    oTSData = oTSData.shuffle(self.DataSet.TSSampleCount)
    oTSData = oTSData.batch(self.BatchSize)
    # Unsafe in Colab with tf 2.9
    #oTSData = oTSData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("Training data feed object:", oTSData)
    return oTSData
  # -----------------------------------------------------------------------------------
  def PreprocessValidationImage(self, p_tImageInVS, p_tLabelInVS):
    tNormalizedImage = self.normalizeImage(p_tImageInVS)
    
    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)
    
    return tNormalizedImage, tTargetOneHot
  # -----------------------------------------------------------------------------------
  def CreateValidationDataFeed(self, p_nBatchSize=None):
    oVSData = tf.data.Dataset.from_tensor_slices((self.DataSet.VSSamples, self.DataSet.VSLabels))
    oVSData = oVSData.map(self.PreprocessValidationImage, num_parallel_calls=8)
    if p_nBatchSize is None:
      p_nBatchSize = self.DataSet.VSSampleCount
    oVSData = oVSData.batch(p_nBatchSize)
    print("Validation data feed object:", oVSData)
    return oVSData
  # --------------------------------------------------------------------------------------
  def _parse_function_augment(self, proto):
      # define your tfrecord again. Remember that you saved your image as a string.
      keys_to_features =              {
                  'image_raw': tf.io.FixedLenFeature([], tf.string),
                  'label': tf.io.FixedLenFeature([], tf.int64) }
      # Load one example
      parsed_features = tf.io.parse_single_example(proto, keys_to_features)
      image = tf.io.decode_raw(parsed_features['image_raw'], tf.uint8)
      image = tf.reshape(image, [32,32,3])
      image = tf.cast(image, tf.float32)
      label = tf.cast(parsed_features['label'], tf.int64)
      label = tf.one_hot(label, 10)
  # --------------------------------------------------------------------------------------      
# =========================================================================================================================


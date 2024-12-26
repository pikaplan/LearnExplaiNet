# ......................................................................................
# MIT License

# Copyright (c) 2020-2023 Pantelis I. Kaplanoglou

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
class KMNISTDataFeed(object):

  # --------------------------------------------------------------------------------------
  def __init__(self, p_oDataSet, p_oConfig, p_nFeatureCount=None, p_bIsRecallingOnly=False):
    super(KMNISTDataFeed, self).__init__()
    # ................................................................

    # // Fields \\
    self.DataSet = p_oDataSet
    self.FileStore = p_oDataSet.fs
    self.PredictBatchSize = p_oConfig["Prediction.BatchSize"]
    self.TrainingBatchSize = p_oConfig["Training.BatchSize"]

    oDataSetStats = self.FileStore.Deserialize("%s-meanstd.pkl" % self.DataSet.Name)
    if oDataSetStats is not None:
      self.PixelMean = oDataSetStats["mean"]
      self.PixelStd = oDataSetStats["std"]
    else:
      self.calculateAndSaveDatasetStats()

    self.ClassCount = p_oConfig["ClassCount"]
    self.InputShape = p_oConfig["InputShape"]
    self.PaddingOffset = 3
    self.PaddingTarget = self.InputShape[0] + 3

    if self.ClassCount is None:
      self.ClassCount = self.DataSet.ClassCount

    if p_bIsRecallingOnly:
      self.TSFeed = None
      self.TSRecallFeed = self.CreateValidationDataFeed((self.DataSet.TSSampleIDs,
                                                         self.DataSet.TSSamples,
                                                         self.DataSet.TSLabels), self.PredictBatchSize)
      self.VSFeed = self.CreateValidationDataFeed((self.DataSet.VSSampleIDs,
                                                   self.DataSet.VSSamples,
                                                   self.DataSet.VSLabels), self.PredictBatchSize)
    else:
      self.TSFeed = self.CreateTrainingDataFeed((self.DataSet.TSSamples
                                                 , self.DataSet.TSLabels), self.TrainingBatchSize)
      self.TSRecallFeed = None
      self.VSFeed = self.CreateValidationDataFeed((self.DataSet.VSSamples,
                                                   self.DataSet.VSLabels), 100)
    # ................................................................

  # --------------------------------------------------------------------------------------
  def calculateAndSaveDatasetStats(self):
    self.PixelMean = np.mean(self.DataSet.TSSamples, axis=(0, 1, 2))
    self.PixelStd = np.std(self.DataSet.TSSamples, axis=(0, 1, 2))
    oDataSetStats = {"mean": self.PixelMean, "std": self.PixelStd}
    self.FileStore.Serialize("%s-meanstd.pkl" % self.DataSet.Name, oDataSetStats)

  # --------------------------------------------------------------------------------------------------------
  def denormalize_image(self, normed_image):
    image = (normed_image * self.PixelStd) + self.PixelMean
    image = image.astype(np.uint8)
    return image

  # --------------------------------------------------------------------------------------------------------
  def normalizeImage(self, p_nImage):
    normed_image = (p_nImage - self.PixelMean) / self.PixelStd
    return normed_image
  # --------------------------------------------------------------------------------------------------------

  def randomCrop(self, image):
    # CIFAR: We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side,
    # and a  32×32  crop is  randomly  sampled  from  the  paddedimage or its horizontal flip.

    # distorted_image = image
    distorted_image = tf.image.pad_to_bounding_box(image, self.PaddingOffset, self.PaddingOffset
                                                   , self.PaddingTarget,
                                                   self.PaddingTarget)  # pad 3 pixels to each side
    distorted_image = tf.image.random_crop(distorted_image, self.InputShape)
    # [WARNING] Flip is a non-label preserving transformation for ExShapes
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    return distorted_image
  # -----------------------------------------------------------------------------------
  @tf.function
  def PreprocessTrainingImageAugmentDataset(self, p_tImageInTS, p_tLabelInTS):
    tImage = tf.cast(p_tImageInTS, tf.float32)  # //[BF] overflow of standardization
    tNormalizedImage = self.normalizeImage(tImage)
    tNewRandomImage = self.randomCrop(tNormalizedImage)

    tTargetOneHot = tf.one_hot(p_tLabelInTS, self.ClassCount)

    return tNewRandomImage, tTargetOneHot

  # --------------------------------------------------------------------------------------
  def CreateTrainingDataFeed(self, p_oDataTuple, p_nBatchSize):
    oTSData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
    oTSData = oTSData.map(self.PreprocessTrainingImageAugmentDataset, num_parallel_calls=8)
    # oTSData = oTSData.cache() # This reduced accuracy on cifar10
    oTSData = oTSData.shuffle(self.DataSet.TSSampleCount)
    oTSData = oTSData.batch(p_nBatchSize)
    # oTSData = oTSData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("Training data feed object:", oTSData)
    return oTSData

  # -----------------------------------------------------------------------------------
  @tf.function
  def PreprocessValidationImageWithID(self, p_tImageID, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)  # //[BF] overflow of standardization
    tNormalizedImage = self.normalizeImage(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return (p_tImageID, tNormalizedImage), tTargetOneHot
    # -----------------------------------------------------------------------------------

  @tf.function
  def PreprocessValidationImage(self, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)  # //[BF] overflow of standardization
    tNormalizedImage = self.normalizeImage(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return tNormalizedImage, tTargetOneHot

  # -----------------------------------------------------------------------------------
  def CreateValidationDataFeed(self, p_oDataTuple, p_nBatchSize=None):
    nArgsCount = len(list(p_oDataTuple))
    if nArgsCount == 2:
      oData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
      oData = oData.map(self.PreprocessValidationImage, num_parallel_calls=8)
    elif nArgsCount == 3:
      oData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
      oData = oData.map(self.PreprocessValidationImageWithID, num_parallel_calls=8)

    if p_nBatchSize is None:
      p_nBatchSize = self.DataSet.VSSampleCount
    oData = oData.batch(p_nBatchSize)
    print("Validation data feed object:", oData)
    return oData
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


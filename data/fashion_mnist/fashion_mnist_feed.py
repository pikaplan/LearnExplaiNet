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
from keras.preprocessing.image import ImageDataGenerator
from keras import preprocessing



# =========================================================================================================================
class FashionMNISTDataFeed(object):

  # --------------------------------------------------------------------------------------
  def __init__(self, dataset_obj, config, feature_count=None, is_recalling_only=False):
    super(FashionMNISTDataFeed, self).__init__()
    # ................................................................

    # // Fields \\
    self.DataSet = dataset_obj
    self.FileStore = dataset_obj.fs
    self.PredictBatchSize = config["Prediction.BatchSize"]
    self.TrainingBatchSize = config["Training.BatchSize"]
    self.datagen = None


    oDataSetStats = self.FileStore.Deserialize("%s-meanstd.pkl" % self.DataSet.Name)
    if oDataSetStats is not None:
      self.PixelMean = oDataSetStats["mean"]
      self.PixelStd = oDataSetStats["std"]
    else:
      self.calculate_and_save_dataset_stats()

    self.ClassCount = config["ClassCount"]
    self.InputShape = config["InputShape"]
    self.PaddingOffset = 3
    self.PaddingTarget = self.InputShape[0] + 3

    if self.ClassCount is None:
      self.ClassCount = self.DataSet.ClassCount

    #self.random_rotation = tf.keras.layers.RandomRotation(factor=(-0.06, 0.06), fill_mode='reflect', interpolation='bilinear', seed=2024, fill_value=0.0)
    #self.random_contrast = tf.keras.layers.RandomContrast(factor=0.2, seed=2024)

    if is_recalling_only:
      self.TSFeed = None
      self.TSRecallFeed = self.create_validation_data_feed((self.DataSet.TSSampleIDs,
                                                            self.DataSet.TSSamples,
                                                            self.DataSet.TSLabels), self.PredictBatchSize)
      self.VSFeed = self.create_validation_data_feed((self.DataSet.VSSampleIDs,
                                                      self.DataSet.VSSamples,
                                                      self.DataSet.VSLabels), self.PredictBatchSize)
    else:
      self.TSFeed = self.create_training_data_feed((self.DataSet.TSSamples
                                                 , self.DataSet.TSLabels), self.TrainingBatchSize)
      self.TSRecallFeed = None
      self.VSFeed = self.create_validation_data_feed((self.DataSet.VSSamples,
                                                      self.DataSet.VSLabels), 100)
    # ................................................................
  # --------------------------------------------------------------------------------------
  def calculate_and_save_dataset_stats(self):
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
  def normalize_image(self, p_nImage):
    normed_image = (p_nImage - self.PixelMean) / self.PixelStd
    return normed_image
  # -----------------------------------------------------------------------------------
  @tf.function
  def preprocess_training_image_augment_dataset(self, p_tImageInTS, p_tLabelInTS):
    tImage = p_tImageInTS
    # tImage = self.random_contrast(tImage)
    tImage = tf.cast(tImage, tf.float32) #//[BF] overflow of standardization
    tImage = self.normalize_image(tImage)

    tImage = tf.image.random_flip_left_right(tImage)
    #tImage = self.random_rotation(tImage)
    tImage = tf.image.pad_to_bounding_box(tImage, self.PaddingOffset, self.PaddingOffset
                                                , self.PaddingTarget, self.PaddingTarget)  # pad 3 pixels to each side
    tImage = tf.image.random_crop(tImage, self.InputShape)

    tTargetOneHot = tf.one_hot(p_tLabelInTS, self.ClassCount)

    return tImage, tTargetOneHot
  # --------------------------------------------------------------------------------------
  def create_training_data_feed(self, p_oDataTuple, p_nBatchSize):
    oTSData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
    oTSData = oTSData.map(self.preprocess_training_image_augment_dataset, num_parallel_calls=8)
    # oTSData = oTSData.cache() # This reduced accuracy on cifar10
    oTSData = oTSData.shuffle(self.DataSet.TSSampleCount)
    oTSData = oTSData.batch(p_nBatchSize)

    # oTSData = oTSData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Training data feed object:", oTSData)
    return oTSData

  # -----------------------------------------------------------------------------------
  def preprocess_validation_image_with_id(self, p_tImageID, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)
    tNormalizedImage = self.normalize_image(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return (p_tImageID, tNormalizedImage), tTargetOneHot
  # -----------------------------------------------------------------------------------
  def preprocess_validation_image(self, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)
    tNormalizedImage = self.normalize_image(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return tNormalizedImage, tTargetOneHot
  # -----------------------------------------------------------------------------------
  def create_validation_data_feed(self, p_oDataTuple, p_nBatchSize=None):
    nArgsCount = len(list(p_oDataTuple))
    if nArgsCount == 2:
      oData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
      oData = oData.map(self.preprocess_validation_image, num_parallel_calls=8)
    elif nArgsCount == 3:
      oData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
      oData = oData.map(self.preprocess_validation_image_with_id, num_parallel_calls=8)

    if p_nBatchSize is None:
      p_nBatchSize = self.DataSet.VSSampleCount
    oData = oData.batch(p_nBatchSize)
    print("Validation data feed object:", oData)
    return oData
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


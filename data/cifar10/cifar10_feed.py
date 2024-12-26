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


import numpy as np
import tensorflow as tf

# =========================================================================================================================
class CCIFAR10DataFeed(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, datasets, config, feature_count=None, is_recalling_only=False):
    super(CCIFAR10DataFeed, self).__init__()
    # ................................................................

    # // Fields \\
    self.DataSet    = datasets
    self.FileStore  = datasets.fs
    self.validation_batchsize = 100# config["Validation.BatchSize"]
    self.PredictBatchSize = config["Prediction.BatchSize"]
    self.TrainingBatchSize = config["Training.BatchSize"]

    # self.DATA_FOLDER = r"T:\Code\Apex\TFResNetCIFAR\data\cifar10"

    # Mean and Std for Z-Score standardization
    oDataSetStats = self.FileStore.obj.load("%s-meanstd.pkl" % self.DataSet.Name)

    if oDataSetStats is not None:
      self.PixelMean = oDataSetStats["mean"]
      self.PixelStd = oDataSetStats["std"]
    else:
      self.calculateAndSaveDatasetStats()

    self.ClassCount = config["ClassCount"]
    self.InputShape = config["InputShape"]

    # We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side ...
    self.PaddingOffset = 4
    self.PaddingTarget = self.InputShape[0] + 4

    if self.ClassCount is None:
      self.ClassCount = self.DataSet.ClassCount

    if is_recalling_only:
      self.TSFeed = None
      self.TSRecallFeed = self.create_validation_feeds((self.DataSet.TSSampleIDs,
                                                        self.DataSet.TSSamples,
                                                        self.DataSet.TSLabels), self.PredictBatchSize)
      self.VSFeed = self.create_validation_feeds((self.DataSet.VSSampleIDs,
                                                  self.DataSet.VSSamples,
                                                  self.DataSet.VSLabels), self.PredictBatchSize)
    else:
      self.TSFeed = self.create_training_feed((self.DataSet.TSSamples
                                                 , self.DataSet.TSLabels), self.TrainingBatchSize)
      self.TSRecallFeed = None
      self.VSFeed = self.create_validation_feeds((self.DataSet.VSSamples,
                                                  self.DataSet.VSLabels), self.validation_batchsize)
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
  @tf.function
  def _normalize_image_tf(self, t_image):
    tNormed = (t_image - tf.constant(self.PixelMean, dtype=tf.float32)) / tf.constant(self.PixelStd, dtype=tf.float32)
    return tNormed
  # --------------------------------------------------------------------------------------------------------
  @tf.function
  def _random_distort_image(self, image):
      # We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side, 
      # and a  32×32  crop is  randomly  sampled  from  the  paddedimage or its horizontal flip.  
    
      #distorted_image = image
      distorted_image = tf.image.pad_to_bounding_box(image, self.PaddingOffset, self.PaddingOffset
                                                   , self.PaddingTarget,
                                                   self.PaddingTarget)
      distorted_image = tf.image.random_crop(distorted_image, self.InputShape)
      distorted_image = tf.image.random_flip_left_right(distorted_image)
      return distorted_image
  # -----------------------------------------------------------------------------------
  @tf.function
  def preprocess_training_image_augment_dataset(self, p_tImageInTS, p_tLabelInTS):
    tImage = tf.cast(p_tImageInTS, tf.float32)  # //[BF] overflow of standardization
    tImage = self._normalize_image_tf(tImage)
    tNewRandomImage = self._random_distort_image(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInTS, self.ClassCount)

    return tNewRandomImage, tTargetOneHot

  # --------------------------------------------------------------------------------------
  def create_training_feed(self, p_oDataTuple, p_nBatchSize):
    oTSData = tf.data.Dataset.from_tensor_slices(p_oDataTuple)
    oTSData = oTSData.map(self.preprocess_training_image_augment_dataset, num_parallel_calls=8)

    #oTSData = oTSData.cache() # This reduces accuracy
    oTSData = oTSData.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = oTSData.shuffle(buffer_size=128 * 10)
    #oTSData = oTSData.shuffle(self.DataSet.TSSampleCount)

    oTSData = oTSData.batch(p_nBatchSize)
    # oTSData = oTSData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("Training data feed object:", oTSData)
    return oTSData

  # -----------------------------------------------------------------------------------
  @tf.function
  def preprocess_validation_image_with_id(self, p_tImageID, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)  # //[BF] overflow of standardization
    tImage = self._normalize_image_tf(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return (p_tImageID, tImage), tTargetOneHot
  # -----------------------------------------------------------------------------------
  @tf.function
  def preprocess_validation_image(self, p_tImageInVS, p_tLabelInVS):
    tImage = tf.cast(p_tImageInVS, tf.float32)  # //[BF] overflow of standardization
    tImage = self._normalize_image_tf(tImage)

    tTargetOneHot = tf.one_hot(p_tLabelInVS, self.ClassCount)

    return tImage, tTargetOneHot

  # -----------------------------------------------------------------------------------
  def create_validation_feeds(self, p_oDataTuple, p_nBatchSize=None):
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


  # --------------------------------------------------------------------------------------------------------
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
  
      train_image = image
      train_image = self.__normalize_image(train_image)
      train_image = self._random_distort_image(train_image)
      
      #train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=self.TrainBatchSize, num_threads=4, capacity=50000, min_after_dequeue=1000)
          
      return train_image, label
    
  # --------------------------------------------------------------------------------------------------------
  def _parse_function(self, proto):
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
  
      train_image = image
      train_image = self.__normalize_image(train_image)
      #train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=self.TrainBatchSize, num_threads=4, capacity=50000, min_after_dequeue=1000)
          
      return train_image, label
  # --------------------------------------------------------------------------------------------------------
  def create_dataset_from_tfrecords(self, filepath, batch_size, p_bIsRepeating, p_bIsPrefetch=True):
      #with tf.device('/cpu:0'):
      # This works with arrays as well
      dataset = tf.data.TFRecordDataset(filepath)
      
      # This dataset will go on forever
      
      if p_bIsRepeating:
          # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
          dataset = dataset.map(self._parse_function_augment, num_parallel_calls=8)
      
          #dataset = dataset.cache() # This reduces accuracy
          dataset = dataset.repeat()
          
          # Set the number of datapoints you want to load and shuffle 
          dataset = dataset.shuffle(buffer_size=128*10)
      else:
          # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
          dataset = dataset.map(self._parse_function, num_parallel_calls=8)
  
          
      # Set the batchsize
      dataset = dataset.batch(batch_size)
      
      if p_bIsPrefetch:
          dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
      #iterator = iter(dataset)
      #image, label = iterator.get_next()
  
      
      # Create an iterator
      #iterator = dataset.__iter__()
      # Create your tf representation of the iterator
      #image, label = next(iterator)
  
      
      return dataset
  # --------------------------------------------------------------------------------------
# =========================================================================================================================  
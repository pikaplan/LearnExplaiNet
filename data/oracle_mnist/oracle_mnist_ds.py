import os
import numpy as np
import gzip
from mllib import FileStore
from mllib.data import CDataSetBase
from keras.datasets import fashion_mnist
from mllib import FileStore

class OracleMNISTDataset(CDataSetBase):
  # --------------------------------------------------------------------------------------
  def __init__(self, datasets_folder, variation=""):
    super(OracleMNISTDataset, self).__init__(p_sName="omnist")  #Oracle characters are the oldest hieroglyphs in China

    self.Variation  = variation
    self.ClassCount = 10
    self.class_names = dict()

    self.fs = FileStore(datasets_folder)
    self.fs = self.fs.subfs(self.Name + self.Variation)
    '''
    dClassMap = self.fs.csv.load("kmnist_classmap.csv")
    print(dClassMap.iterrows())

    # load these nice japanese characters as class names
    for nIndex, dRow in dClassMap.iterrows():
      self.class_names[int(dRow["index"])] = dRow["char"]
    '''
  # --------------------------------------------------------------------------------------
  def load(self):
    if not self.LoadCache(self.fs, p_sTargetsFilePrefix="Labels"):
      nTSSamples, nTSLabels = self.prepare("train")
      self.TrainingSet(nTSSamples, nTSLabels)
      nVSSamples, nVSLabels = self.prepare("t10k")
      self.ValidationSet(nVSSamples, nVSLabels)
      self.SaveCache(self.fs, p_sTargetsFilePrefix="Labels")
  # --------------------------------------------------------------------------------------
  def prepare(self, kind):
    """Load Oracle-MNIST data from `path`"""
    nLabelsFileName = os.path.join(self.fs.base_folder, '%s-labels-idx1-ubyte.gz' % kind)
    nSamplesFileName = os.path.join(self.fs.base_folder, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(nLabelsFileName, "rb") as oFile:
      nLabels = np.frombuffer(oFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(nSamplesFileName, "rb") as oFile:
      nSamples = np.frombuffer(oFile.read(), dtype=np.uint8, offset=16)

    nSamples = nSamples.reshape(len(nLabels), 28, 28, 1)
    return nSamples, nLabels
  # --------------------------------------------------------------------------------------



if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from mllib.system.backends import TensorflowBackend

  TensorflowBackend.info()
  TensorflowBackend.debug()

  oDS = OracleMNISTDataset(r"C:\MLData")
  oDS.load()
  oDS.info()



  if False:
    # Look at some sample images from dataset
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(oDS.TSSamples[i].squeeze(), cmap='gray')
      plt.title(f"Label: {oDS.TSLabels[i]}")
      plt.axis('off')
    plt.show()

  # Pipeline
  if False:
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    #https: // www.tensorflow.org / api_docs / python / tf / keras / layers / RandomContrast
    oRotation = tf.keras.layers.RandomRotation( factor=(-0.12, 0.12),
                    fill_mode='reflect', interpolation='bilinear', seed=2024, fill_value=0.0)
    oContrast = tf.keras.layers.RandomContrast( factor=0.4, seed=2024)

    @tf.function
    def augment(image, label):
      tImage = image
      tImage = tf.image.pad_to_bounding_box(tImage, 3, 3, 31, 31)  # pad 3 pixels to each side
      tImage = oRotation(tImage)
      tImage = oContrast(tImage)
      tLabel = tf.one_hot(label, 10)
      return     tImage, tLabel


    oTSData = tf.data.Dataset.from_tensor_slices((oDS.TSSamples, oDS.TSSampleIDs))
    oTSData = oTSData.map(augment, num_parallel_calls=8)
    oTSData = oTSData.cache()
    oTSData = oTSData.shuffle(oDS.TSSampleCount)
    oTSData = oTSData.batch(128)

    (nSamples, nLabels) =  next(iter(oTSData))
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(nSamples[i].numpy().squeeze(), cmap='gray')
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()


  # Feeded
  if True:
    from data.oracle_mnist.oracle_mnist_feed import OracleMNISTDataFeed
    config = {"InputShape": [28,28,1], "ClassCount":10, "Training.BatchSize": 128, "Validation.BatchSize": 128, "Prediction.BatchSize": 200}
    oFeed = OracleMNISTDataFeed(oDS, config)
    print(f"mean: {oFeed.PixelMean} std:{oFeed.PixelStd}")
    (nSamples, nLabels) =  next(iter(oFeed.TSFeed))
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(nSamples[i].numpy().squeeze(), cmap='gray')
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()


    # Look at some sample images from dataset
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      nImage = oDS.TSSamples[i].squeeze()
      nImageNormed = oFeed.normalizeImage(nImage)
      #nImageNormed = (nImage - oFeed.PixelMean)/oFeed.PixelStd
      plt.imshow(nImageNormed, cmap='gray')
      plt.title(f"Label: {oDS.TSLabels[i]}")
      plt.axis('off')
    plt.show()



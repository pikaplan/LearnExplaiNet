import keras.preprocessing.image
import numpy as np

from mllib import FileStore
from mllib.data import CDataSetBase
from keras.datasets import fashion_mnist
from mllib import FileStore

class FashionMNISTDataset(CDataSetBase):
  # --------------------------------------------------------------------------------------
  def __init__(self, datasets_folder, variation=""):
    super(FashionMNISTDataset, self).__init__(p_sName="fmnist")

    self.Variation  = variation
    self.ClassCount = 10
    self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    self.fs = FileStore(datasets_folder)
    self.fs = self.fs.subfs(self.Name + self.Variation)
  # --------------------------------------------------------------------------------------
  def load(self):
    if not self.LoadCache(self.fs, p_sTargetsFilePrefix="Labels"):
      self.prepare()
      self.SaveCache(self.fs, p_sTargetsFilePrefix="Labels")
  # --------------------------------------------------------------------------------------
  def prepare(self, is_overwriting=False):
    (oTSImages, oTSLabels), (oVSImages, oVSLabels) = fashion_mnist.load_data()#data_dir=self.fs.subfs("downloaded").base_folder)
    self.TrainingSet(oTSImages[...,np.newaxis], oTSLabels)
    self.ValidationSet(oVSImages[...,np.newaxis], oVSLabels)
  # --------------------------------------------------------------------------------------


if __name__ == "__main__":
  oDS = FashionMNISTDataset(r"C:\MLData\FASHION_MNIST")
  oDS.load()
  oDS.PrintInfo()


  import matplotlib.pyplot as plt

  if False:
    # Look at some sample images from dataset
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(oDS.TSSamples[i].squeeze(), cmap='gray')
      plt.title(f"Label: {oDS.TSLabels[i]}")
      plt.axis('off')
    plt.show()

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


  if True:
    from data.fashion_mnist.fashion_mnist_feed import FashionMNISTDataFeed
    config = {"InputShape": [28,28,1], "ClassCount":10, "Training.BatchSize": 128, "Validation.BatchSize": 128, "Prediction.BatchSize": 200}
    oFeed = FashionMNISTDataFeed(oDS, config)
    print(f"mean: {oFeed.PixelMean} std:{oFeed.PixelStd}")
    (nSamples, nLabels) =  next(iter(oFeed.TSFeed))
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(nSamples[i].numpy().squeeze(), cmap='gray')
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()



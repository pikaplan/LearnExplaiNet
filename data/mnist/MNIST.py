import os
import pandas as pd
from mllib.data import CDataSetBase 
import numpy as np
from mllib import CMLSystem, CFileStore

# =========================================================================================================================
class CMNISTDataSet(CDataSetBase):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_oDatasetsFS, p_sVariation=""):
    super(CMNISTDataSet, self).__init__(p_sName="mnist")
    self.Variation  = p_sVariation
    self.ClassCount = 10
   
    self.FileStore = p_oDatasetsFS.SubFS(self.Name + self.Variation)
  # --------------------------------------------------------------------------------------
  def Load(self):    
    if not self.LoadCache(self.FileStore):
      self.Prepare(True)

  # --------------------------------------------------------------------------------------
  def load(self):
    self.Load()
  # --------------------------------------------------------------------------------------
  def Prepare(self, p_bIsOverwritting=False):
    if p_bIsOverwritting or (not self.FileStore.Exists("Samples.pkl")):
   
      dTrain  = pd.read_csv(self.FileStore.File("mnist_train.csv"), header=0)
      nArray = dTrain.to_numpy()
      nSampleCount = nArray.shape[0]
      self.TrainingSet(nArray[:,1:].reshape(nSampleCount, 28, 28, 1) , nArray[:,0])
  
      dTest   = pd.read_csv(self.FileStore.File("mnist_test.csv"), header=0)
      nArray = dTest.to_numpy()
      nSampleCount = nArray.shape[0]
      self.ValidationSet(nArray[:,1:].reshape(nSampleCount, 28, 28, 1), nArray[:,0])
          
      self.Samples = np.concatenate([self.TSSamples, self.VSSamples], axis=0)
      self.Labels  = np.concatenate([self.TSLabels, self.VSLabels], axis=0)
              
      if p_bIsOverwritting:
        print("Is overwritting")
      self.SaveCache(self.FileStore)
  # --------------------------------------------------------------------------------------        
# =========================================================================================================================


if __name__ == "__main__":
  import mllib as ml
  import matplotlib.pyplot as plt
  
  os.chdir("..\..")
  
  ml.CMLSystem.DATASET_FOLDER = "C:\\MLData"
  #ml.CMLSystem.MODEL_FOLDER   = "T:\\MLModels.Keep"
  #ml.CMLSystem.RANDOM_SEED    = 2022
  #oSys = ml.CMLSystem.Instance() 
  oDatasetsFS = ml.FileStore(ml.CMLSystem.DATASET_FOLDER)

  oMNIST = CMNISTDataSet(oDatasetsFS)
  oMNIST.Load()
  print("[MNIST] Total Samples:%d   | Features:%d | Classes: %d" % (oMNIST.SampleCount, oMNIST.FeatureCount, oMNIST.ClassCount))
  print("[MNIST] Training:%d        |" % (oMNIST.TSSampleCount))
  print("[MNIST] MemoryTest:%d            |" % (oMNIST.VSSampleCount))


  if True:
    from data.mnist.MNISTDataFeed import CMNISTDataFeed
    config = {"InputShape": [28,28,1], "ClassCount":10, "Training.BatchSize": 128, "Validation.BatchSize": 128, "Prediction.BatchSize": 200}
    oFeed = CMNISTDataFeed(oMNIST, config)
    print(f"mean: {oFeed.PixelMean} std:{oFeed.PixelStd}")
    (nSamples, nLabels) =  next(iter(oFeed.TSFeed))
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(nSamples[i].numpy().squeeze(), cmap='gray')
      plt.title(f"Label: {nLabels[i]}")
      plt.axis('off')
    plt.show()


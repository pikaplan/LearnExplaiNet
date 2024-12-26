import numpy as np


# =========================================================================================================================
class CLayerActivationCollection(object):
  # --------------------------------------------------------------------------------------
  def __init__(self):
    self.Layers = dict()
    self.CurrentSample = None
    self.LayerIndex = None

  # --------------------------------------------------------------------------------------
  def NewSample(self):
    self.LayerIndex = 0

  # --------------------------------------------------------------------------------------
  def append(self, p_nActivationTensor):
    if self.LayerIndex not in self.Layers:
      self.Layers[self.LayerIndex] = []

    # print("a[%d]: %s" % (self.LayerIndex, p_nActivationTensor.shape))

    self.Layers[self.LayerIndex].append(p_nActivationTensor)
    self.LayerIndex += 1
    # print(self.LayerIndex)

  # --------------------------------------------------------------------------------------
  def appendModule(self, p_oLayerActivationCollection):
    for sKey in p_oLayerActivationCollection.Layers.keys():
      oActivationsOfLayer = p_oLayerActivationCollection.Layers[sKey]
      if self.LayerIndex not in self.Layers:
        self.Layers[self.LayerIndex] = []

      nCurrentIndex = len(self.Layers[self.LayerIndex])

      self.Layers[self.LayerIndex].append(oActivationsOfLayer[nCurrentIndex])
      self.LayerIndex += 1
      # print(self.LayerIndex)

  # --------------------------------------------------------------------------------------
  def LayerAsNumpy(self, p_nKey):
    print("Activations for module %d" % p_nKey)
    oBatchActivationList = self.Layers[p_nKey]
    nActivations = np.concatenate(oBatchActivationList, axis=0)
    return nActivations

    ''''
    nBatchCount = len(oBatchActivationsOfLayer)
    nBatchActivationShape = list(oBatchActivationsOfLayer[0].shape)
    nBatchSize  = nBatchActivationShape[0]


    nResult = np.zeros([nBatchCount*nBatchSize] + nBatchActivationShape[1:], np.float32)
    for nIndex,nBatchActivations in enumerate(oBatchActivationsOfLayer):
      nResult[(nIndex*nBatchSize):(nIndex*nBatchSize + nBatchSize),:,:] = nBatchActivations[:,:,:]

    #print(nResult.shape)
    return nResult
    '''

  # --------------------------------------------------------------------------------------
  def __iter__(self):
    self.__keys = list(self.Layers.keys())
    self.__index = 0
    return self

  # --------------------------------------------------------------------------------------
  def __next__(self):
    if self.__index < len(self.__keys):
      nKey = self.__keys[self.__index]
      oBatchActivationList = self.Layers[nKey]
      nActivations = np.concatenate(oBatchActivationList, axis=0)

      self.__index += 1
      return nKey, nActivations
    else:
      raise StopIteration
  # --------------------------------------------------------------------------------------

# =========================================================================================================================


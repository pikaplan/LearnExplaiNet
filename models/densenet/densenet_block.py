import math
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras.layers import (
    Conv2D, Dense, BatchNormalization, Activation, Concatenate, ZeroPadding2D,
    Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
  )
from keras.regularizers import L2


# ======================================================================================================================
class DenseNetBlock(layers.Layer):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, config, module_count):
    super(DenseNetBlock, self).__init__()
    
    self.cfg = config
    self.module_count = module_count
    self.growth_features = self.cfg["DenseNet.Growth.Features"]
    self.is_bottleneck   = self.cfg["DenseNet.Module.IsBottleNeck"]
    if "DenseNet.BatchNorm.Momentum" in self.cfg:
      self.batchnorm_momentum = self.cfg["DenseNet.BatchNorm.Momentum"]
    else:
      self.batchnorm_momentum = 0.99
    if "DenseNet.BatchNorm.Epsilon" in self.cfg:
      self.batchnorm_epsilon = self.cfg["DenseNet.BatchNorm.Epsilon"]
    else:
      self.batchnorm_epsilon = 0.001
    self.weight_decay = self.cfg["Training.WeightDecay"]
    
    self.Bottleneck_BN = []
    self.Bottleneck_Conv1x1 = []
    
    self.BatchNorm  = []
    self.Activation = []
    self.PadSpatial = []
    self.Conv2D     = []
    self.Concat     = []
    self.create()
# --------------------------------------------------------------------------------------------------------------------
  def create_module(self, module_index):
    if self.is_bottleneck:
      print(f"        |__ Bottleneck")
      oBottleneck_BN = BatchNormalization(axis=3, momentum=self.batchnorm_momentum,
                                              epsilon=self.batchnorm_epsilon)
      oBottleneck_Conv1x1 = Conv2D(4*self.growth_features, (1,1), (1,1), padding="valid", use_bias=False,
                                      kernel_initializer="he_normal",kernel_regularizer=L2(self.weight_decay))
      self.Bottleneck_BN.append(oBottleneck_BN)                                
      self.Bottleneck_Conv1x1.append(oBottleneck_Conv1x1)

    oBatchNorm = BatchNormalization(axis=3, momentum=self.batchnorm_momentum, 
                                              epsilon=self.batchnorm_epsilon)
    oActivation = Activation('relu')
    oPadSpatial        = ZeroPadding2D((1,1))
    oConv2D = Conv2D(self.growth_features, (3,3), (1,1), padding="valid", use_bias=False, 
                                      kernel_initializer="he_normal",kernel_regularizer=L2(self.weight_decay) )
    self.BatchNorm.append(oBatchNorm)
    self.Activation.append(oActivation)
    self.PadSpatial.append(oPadSpatial)
    self.Conv2D.append(oConv2D)
    #nInputFeatureCount = input.get_shape().as_list()[3]
  # --------------------------------------------------------------------------------------------------------------------
  def create(self):
    for nModuleIndex in range(self.module_count):
      print(f"    |__ Module {nModuleIndex+1}")
      self.create_module(nModuleIndex)
      oConcat = Concatenate(axis=3)
      self.Concat.append(oConcat)
  # --------------------------------------------------------------------------------------------------------------------
  def call(self, input):
    tA = input

    oSkipConnections = [tA]

    for nModuleIndex in range(self.module_count):
      if self.is_bottleneck:
        tA = self.Bottleneck_BN[nModuleIndex](tA)
        tA = self.Bottleneck_Conv1x1[nModuleIndex](tA)

      tA = self.BatchNorm[nModuleIndex](tA)
      tA = self.Activation[nModuleIndex](tA)
      tA = self.PadSpatial[nModuleIndex](tA)
      tA = self.Conv2D[nModuleIndex](tA)
      oSkipConnections.append(tA)
      tA = self.Concat[nModuleIndex](oSkipConnections)
    return tA
  # --------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================






  
# ======================================================================================================================
class DenseNetTransitionModule(layers.Layer):
  def __init__(self, config, input_features, compression=1.0, output_features=None):
    super(DenseNetTransitionModule, self).__init__()
    
    self.cfg = config
    self.input_features = input_features
    self.compression = compression
    self.output_features = output_features
    
    if self.output_features is None:
      self.output_features = self.input_features
      if self.compression < 1.0:
        self.output_features = math.floor(self.input_features*self.compression)

    if "DenseNet.BatchNorm.Momentum" in self.cfg:
      self.batchnorm_momentum = self.cfg["DenseNet.BatchNorm.Momentum"]
    else:
      self.batchnorm_momentum = 0.99
    if "DenseNet.BatchNorm.Epsilon" in self.cfg:
      self.batchnorm_epsilon = self.cfg["DenseNet.BatchNorm.Epsilon"]
    else:
      self.batchnorm_epsilon = 0.001
    self.weight_decay = self.cfg["Training.WeightDecay"]
    
    self.BatchNorm = None
    self.Activation = None
    self.TransConv2D = None
    self.SpatialDownsampling = None
    self.create()

  # --------------------------------------------------------------------------------------------------------------------
  def create(self):
    self.BatchNorm = BatchNormalization(axis=3, momentum=self.batchnorm_momentum, 
                                              epsilon=self.batchnorm_epsilon)
    self.Activation = Activation('relu')
    self.TransConv2D = Conv2D(self.output_features, (1,1), (1,1), padding="valid", use_bias=False, 
                                      kernel_initializer="he_normal",kernel_regularizer=L2(self.weight_decay) )
    self.SpatialDownsampling = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
  # --------------------------------------------------------------------------------------------------------------------
  def call(self, input):
    tA = input
    # //TODO: Infer the input features during build
    #nInputFeatureCount = input.get_shape().as_list()[3]
    tA = self.BatchNorm(tA)
    tA = self.Activation(tA)
    tA = self.TransConv2D(tA)
    tA = self.SpatialDownsampling(tA)
    return tA
# ======================================================================================================================





if __name__ == "__main__":
  from mllib.system.backends import TensorflowBackend
  
  TensorflowBackend.debug()
  nImage = np.random.rand(1, 32,32,3).astype(np.float32)
  
  CONFIG =  {

           "DenseNet.IsTinyImage": True
          ,"DenseNet.Stem.Features": [16]
          ,"DenseNet.Blocks.Modules": [12, 12, 12]
          ,"DenseNet.Blocks.Transition": [False, True, True]
          ,"DenseNet.Growth.Features": 12
          ,"DenseNet.Module.IsBottleNeck": False
          ,"DenseNet.Transition.Compression": 1.0
          ,"DenseNet.BatchNorm.Momentum": 0.9
          , "ClassCount": 10
          , "Training.WeightDecay"		: 1e-4
  }


  CONFIG_2 =  {

           "DenseNet.IsTinyImage": True
          ,"DenseNet.Stem.Features": [24]
          ,"DenseNet.Blocks.Modules": [16, 16, 16]
          ,"DenseNet.Blocks.Transition": [False, True, True]
          ,"DenseNet.Growth.Features": 12
          ,"DenseNet.Module.IsBottleNeck": True
          ,"DenseNet.Transition.Compression": 0.5
          ,"DenseNet.BatchNorm.Momentum": 0.9
          , "ClassCount": 10
          , "Training.WeightDecay"		: 1e-4
  }


  
  
  oModule = DenseNetBlock(CONFIG_2, 32)
  #oModule = DenseNetTransitionModule(CONFIG, 24)
  tResult = oModule(nImage)
  print(tResult.numpy().shape)

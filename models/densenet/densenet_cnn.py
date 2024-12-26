import numpy as np
import tensorflow as tf
import keras
from keras.layers import (
    Conv2D, Dense, BatchNormalization, Activation, MaxPooling2D, 
    Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
  )
from keras.regularizers import L2

from models.densenet.densenet_block import DenseNetBlock, DenseNetTransitionModule



# ======================================================================================================================
class DenseNet(keras.Model):
  DenseModuleClass = DenseNetBlock
  DenseTransitionClass = DenseNetTransitionModule



  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, config):
    super(DenseNet, self).__init__()
    self.cfg = config
    self.is_classifier = ("ClassCount" in self.cfg)
    if self.is_classifier:
      self.class_count = self.cfg["ClassCount"]
    self.is_tiny_image = self.cfg["DenseNet.IsTinyImage"]
    self.growth_features = self.cfg["DenseNet.Growth.Features"]
    if "DenseNet.Transition.Compression" in self.cfg:
      self.compression = self.cfg["DenseNet.Transition.Compression"]
    else:
      self.compression = 1.0

    self.is_bottleneck = self.cfg["DenseNet.Module.IsBottleNeck"]
    if "DenseNet.BatchNorm.Momentum" in self.cfg:
      self.batchnorm_momentum = self.cfg["DenseNet.BatchNorm.Momentum"]
    else:
      self.batchnorm_momentum = 0.99
    if "DenseNet.BatchNorm.Epsilon" in self.cfg:
      self.batchnorm_epsilon = self.cfg["DenseNet.BatchNorm.Epsilon"]
    else:
      self.batchnorm_epsilon = 0.001
    self.weight_decay = self.cfg["Training.WeightDecay"]

    self.block_setup = self.cfg["DenseNet.Blocks.Modules"]
    if "DenseNet.Blocks.Transition" in self.cfg:
      self.transition_setup = self.cfg["DenseNet.Blocks.Transition"]
    else:
      self.transition_setup = [True]*len(self.block_setup)
      self.transition_setup[0] = False

    if "DenseNet.Stem.Features" in self.cfg:
      self.stem_features = self.cfg["DenseNet.Stem.Features"]
    elif self.is_tiny_image:
      self.stem_features = [16]
    else:
      self.stem_feature = self.growth_features*2

    self.StemConv       = None 
    self.StemMaxPool    = None
    self.blocks = []
    self.transitions = [] 

    self.LastBatchNorm = None
    self.LastActivation = None
    self.GlobalPool = None
    self.Logits = None
    self.Classes = None  
    
    self.create()
  # --------------------------------------------------------------------------------------------------------------------
  def create(self):
    if self.is_tiny_image:
      self.StemConv = Conv2D(self.stem_features[0], (3,3), strides=(1,1), padding="same", use_bias=False,
                  kernel_initializer="he_normal", kernel_regularizer=L2(self.weight_decay))                 
    else:
      # //TODO: Check if bias is used in ImageNet architecture
      self.StemConv = Conv2D(self.stem_features[0], (7,7), strides=(2,2), padding="same", use_bias=False,
                  kernel_initializer="he_normal", kernel_regularizer=L2(self.weight_decay)) 
      self.StemMaxPool = MaxPooling2D((3,3), strides=(2,2))

    # Modules stack ending with a transition layer
    nFeatures = self.stem_features[0]
    for nBlockIndex, nModulesOfBlock in enumerate(self.block_setup):
      if self.transition_setup[nBlockIndex]:
        oTransition = DenseNet.DenseTransitionClass(self.cfg, nFeatures, self.compression)
        self.transitions.append(oTransition)
        print(f"    |__ {DenseNet.DenseTransitionClass.__name__} | input:{oTransition.input_features} compression:{oTransition.compression:1f} output:{oTransition.output_features}")
        nFeatures = oTransition.output_features

      print(f" |__ {DenseNet.DenseModuleClass.__name__} {nBlockIndex+1}")
      oBlock = DenseNet.DenseModuleClass(self.cfg, nModulesOfBlock)
      nFeatures += nModulesOfBlock*self.growth_features
      print(f" |__ Block Output Features:{nFeatures}")
      self.blocks.append(oBlock)
        
    self.LastBatchNorm = BatchNormalization(axis=3, momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.LastActivation = Activation("relu")
  
    print("Global Average Pooling")
    self.GlobalPool = GlobalAveragePooling2D()
    
    if self.is_classifier:
        self.Logits = Dense(self.class_count, bias_initializer="zeros", kernel_regularizer=L2(self.weight_decay))
        self.Classes = Softmax()
  # --------------------------------------------------------------------------------------------------------------------
  def call(self, input):
    tA = self.StemConv(input)
    if not self.is_tiny_image:
      tA = self.StemMaxPool(tA)

    for nBlockIndex, oBlock in enumerate(self.blocks):
      if self.transition_setup[nBlockIndex]:
        tA = self.transitions[nBlockIndex - 1](tA)
      tA = self.blocks[nBlockIndex](tA)
      
    tA = self.LastBatchNorm(tA)
    tA = self.LastActivation(tA)
    tA = self.GlobalPool(tA)
    if self.is_classifier:
      tA = self.Logits(tA)
      tA = self.Classes(tA)
      
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

  
  oModule = DenseNet(CONFIG_2)
  tResult = oModule(nImage)
  print(tResult.numpy().shape)
  oModule.summary(expand_nested=True)


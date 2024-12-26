from tensorflow import keras

import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras.layers import (
  Conv2D, Dense, BatchNormalization, Activation,
  Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
)
from keras.regularizers import L2

from .residual_module_constants import ResConnectionType, ResModuleType
from mllib.utils import DefaultSetting


# =========================================================================================================================
class ResidualModule(layers.Layer):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_nFeatures, p_nStrideOnInput, p_eModuleType, p_nWeightDecay, batchnorm_momentum=0.99,
               batchnorm_epsilon=0.001, p_nResidualConnectionType=ResConnectionType.AVG_POOL_PAD, p_bIsAnalyzing=False):
    super(ResidualModule, self).__init__()

    # ................................................................
    # // Attributes \\
    self.Features = p_nFeatures
    self.StrideOnInput = p_nStrideOnInput
    self.ModuleType = p_eModuleType
    self.ResidualConnectionType = p_nResidualConnectionType
    self.Expansion = 1
    self.WeightDecay = p_nWeightDecay
    # self.DropOutRate    = 0.5
    self.batchnorm_momentum = batchnorm_momentum
    self.batchnorm_epsilon = batchnorm_epsilon

    self.MustCheckToPadFeatures = True

    # // Layers \\
    self.Input = None
    self.InputFeatures = None
    self.SpatialDownSampling = None
    self.Conv1 = None
    self.Conv2 = None
    self.Conv3 = None
    self.Activation = None
    self.Output = None

    self.IsAnalyzing = p_bIsAnalyzing
    self.LayerActivations = []
    # ................................................................
    self.CreateBlock()

  # --------------------------------------------------------------------------------------------------------
  def CreateResidualConnection(self):
    self.MustCheckToPadFeatures = True

    tSpatialDownsampling = None
    if self.ResidualConnectionType == ResConnectionType.AVG_POOL_PAD:
      # Downsample spatially
      if (self.StrideOnInput > 1):
        tSpatialDownsampling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        print("     |__ Average pooling on residual connection input")
    else:
      if (self.StrideOnInput > 1):
        if self.ResidualConnectionType < ResConnectionType.AVG_POOL_PAD:
          nWindowSize = (self.ResidualConnectionType, self.ResidualConnectionType)
          nConvStride = (self.StrideOnInput, self.StrideOnInput)
          tSpatialDownsampling = Conv2D(self.Features, kernel_size=nWindowSize, strides=nConvStride
                                        , padding="same", use_bias=False, kernel_initializer="he_uniform"
                                        , kernel_regularizer=L2(self.WeightDecay))
          print(
            f"     |__ Conv {self.ResidualConnectionType}x{self.ResidualConnectionType}/{self.StrideOnInput} on residual connection input")
        self.MustCheckToPadFeatures = False
    if tSpatialDownsampling is None:
      print("     |__ no adaptation for residual connection input")
    return tSpatialDownsampling

  # --------------------------------------------------------------------------------------------------------
  def PadFeatures(self, p_tX, p_nFeatureDepth, p_sName=None):
    nInputShape = p_tX.shape.as_list()

    nPadFeatures = p_nFeatureDepth - nInputShape[3]
    tPad = tf.pad(p_tX, [[0, 0], [0, 0], [0, 0], [nPadFeatures // 2, nPadFeatures // 2]], name=p_sName)

    return tPad
    # --------------------------------------------------------------------------------------

  def CallResidualConnection(self, p_tInput):
    tResidual = p_tInput
    if self.SpatialDownSampling is not None:
      tResidual = self.SpatialDownSampling(p_tInput)
      if self.MustCheckToPadFeatures and (self.InputFeatures != self.Features):
        tResidual = self.PadFeatures(tResidual, self.Features, p_sName="pad_skip")
      else:
        tResidual = self.SpatialDownSampling(p_tInput)
    else:
      if self.MustCheckToPadFeatures and (self.InputFeatures != self.Features):
        tResidual = self.PadFeatures(tResidual, self.Features, p_sName="pad_skip")

    return tResidual
    # --------------------------------------------------------------------------------------

  def CreateBlock(self):
    self.SpatialDownSampling = self.CreateResidualConnection()
    self.ResidualBN = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.Pad1 = ZeroPadding2D((1, 1))
    self.Conv1 = Conv2D(self.Features, (3, 3), (self.StrideOnInput, self.StrideOnInput), padding="valid",
                        use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=L2(self.WeightDecay))
    self.BN1 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.Pad2 = ZeroPadding2D((1, 1))
    self.Conv2 = Conv2D(self.Features, (3, 3), (1, 1), padding="valid", use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=L2(self.WeightDecay))
    self.BN2 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)

  # --------------------------------------------------------------------------------------
  def __addActivation(self, p_tTensor):
    if self.IsAnalyzing:
      self.LayerActivations.append(p_tTensor.numpy())

  # --------------------------------------------------------------------------------------
  def call(self, p_tInput):
    self.Input = p_tInput
    self.InputFeatures = self.Input.get_shape().as_list()[3]
    x = self.Input

    tResidual = self.CallResidualConnection(x)

    if self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value:
      x = Activation("relu")(x)
    elif self.ModuleType == ResModuleType.FULL_PREACTIVATION.value:
      x = Activation("relu")(self.Parent.BatchNormalization(x))

    a = self.Pad1(x)
    a = self.Conv1(a)
    a = self.BN1(a)
    a = Activation("relu")(a)

    a = self.Pad2(a)
    a = self.Conv2(a)

    if self.ModuleType == ResModuleType.ORIGINAL.value:
      b = self.BN2(a)
      tResidualBN = self.ResidualBN(tResidual)
      y = Activation("relu")(b + tResidualBN)
    elif self.ModuleType == ResModuleType.BN_AFTER_ADDITION.value:
      u = self.BN2(a + tResidual)
      y = Activation("relu")(u)
    elif self.ModuleType == ResModuleType.RELU_BEFORE_ADDITION.value:
      b = Activation("relu")(self.BN2(a))
      y = b + tResidual
    elif self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value:
      b = self.BN2(a)
      y = b + tResidual
    elif self.ModuleType == ResModuleType.FULL_PREACTIVATION.value:
      y = a + tResidual

    self.Output = y

    return self.Output
  # --------------------------------------------------------------------------------------
# =========================================================================================================================

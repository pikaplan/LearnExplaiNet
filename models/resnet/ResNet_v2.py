import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import (
    Conv2D, Dense, BatchNormalization, Activation,
    Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
  )
from keras.regularizers import L2
from mllib.utils import DefaultSetting


# =========================================================================================================================
# --------------------------------------------------------------------------------------------------------
# Types of residual skip connections
# --------------------------------------------------------------------------------------------------------
class ResConnectionType:
    CONV_1x1     = 1
    CONV_2x2     = 2
    CONV_3x3     = 3
    AVG_POOL_PAD = 4
# =========================================================================================================================




# =========================================================================================================================
class ResidualModule(layers.Layer):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, features, stride_on_input, weight_decay, batchnorm_momentum=0.99,
               batchnorm_epsilon=0.001, residual_connection_type=ResConnectionType.AVG_POOL_PAD):
    super(ResidualModule, self).__init__()

    # ................................................................
    # // Attributes \\
    self.Features = features
    self.StrideOnInput = stride_on_input
    self.ResidualConnectionType = residual_connection_type
    self.Expansion = 1
    self.WeightDecay = weight_decay
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
    self.Act1 = None
    self.Act2 = None
    self.Output = None
    # ................................................................
    self.create_block()
  # --------------------------------------------------------------------------------------------------------------------
  def create_residual_connection(self):
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
          nWindowSize = (self.ResidualType, self.ResidualType)
          nConvStride = (self.StrideOnInput, self.StrideOnInput)
          tSpatialDownsampling = Conv2D(self.Features, kernel_size=nWindowSize, strides=nConvStride
                                        , padding="same", use_bias=False, kernel_initializer="he_uniform"
                                        , kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
          print("     |__ Conv2D on residual connection input")
        self.MustCheckToPadFeatures = False
    if tSpatialDownsampling is None:
      print("     |__ no adaptation for residual connection input")
    return tSpatialDownsampling
  # --------------------------------------------------------------------------------------------------------------------
  def create_block(self):
    self.SpatialDownSampling = self.create_residual_connection()
    self.ResidualBN = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.Pad1 = ZeroPadding2D((1, 1))
    self.Conv1 = Conv2D(self.Features, (3, 3), (self.StrideOnInput, self.StrideOnInput), padding="valid",
                        use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
    print(f"     |_ Conv3x3 Features:{self.Features} Stride:{self.StrideOnInput}")
    self.BN1 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.Act1 = Activation("relu")

    self.Pad2 = ZeroPadding2D((1, 1))
    self.Conv2 = Conv2D(self.Features, (3, 3), (1, 1), padding="valid", use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
    print(f"     |_ Conv3x3 Features:{self.Features} Stride:{self.StrideOnInput}")
    self.BN2 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
    self.Act2 = Activation("relu")
  # --------------------------------------------------------------------------------------------------------------------
  def _pad_features(self, p_tX, p_nFeatureDepth, p_sName=None):
    nInputShape = p_tX.shape.as_list()

    nPadFeatures = p_nFeatureDepth - nInputShape[3]
    tPad = tf.pad(p_tX, [[0, 0], [0, 0], [0, 0], [nPadFeatures // 2, nPadFeatures // 2]], name=p_sName)

    return tPad
  # --------------------------------------------------------------------------------------------------------------------
  def _call_residual_connection(self, p_tInput):
    tResidual = p_tInput
    if self.SpatialDownSampling is not None:
      tResidual = self.SpatialDownSampling(p_tInput)
      if self.MustCheckToPadFeatures and (self.InputFeatures != self.Features):
        tResidual = self._pad_features(tResidual, self.Features, p_sName="pad_skip")
      else:
        tResidual = self.SpatialDownSampling(p_tInput)
    else:
      if self.MustCheckToPadFeatures and (self.InputFeatures != self.Features):
        tResidual = self._pad_features(tResidual, self.Features, p_sName="pad_skip")

    return tResidual
  # --------------------------------------------------------------------------------------------------------------------
  def call(self, input):
    self.Input = input
    self.InputFeatures = self.Input.get_shape().as_list()[3]
    tX = self.Input

    tResidual = self._call_residual_connection(tX)
    tA = self.Pad1(tX)
    tA = self.Conv1(tA)
    tA = self.BN1(tA)
    tA = self.Act1(tA)

    tA = self.Pad2(tA)
    tA = self.Conv2(tA)

    tB = self.BN2(tA)
    tY = self.Act2(tB + tResidual)

    self.Output = tY
    return self.Output
  # --------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================






# =========================================================================================================================
class ResNet(keras.Model):
  ResidualModuleClass = ResidualModule

  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, config)  :  # , p_nModuleCount=None, p_oConvFeatures=[], p_oWindowSizes=[], p_oPoolStrides=[], p_bHasBias=True, p_sActivationFunction="relu"):
    super(ResNet, self).__init__()
    # ................................................................
    # // Attributes \\
    # ................................................................
    self.ModuleType             = config["ResNet.ModuleType"]
    self.Features               = config["ResNet.Blocks.Features"]
    self.StackSetup             = config["ResNet.Blocks.Modules"]
    self.StemFeatures           = DefaultSetting(config, "ResNet.Stem.Features", [self.Features[0]])
    self.BatchNormMomentum      = DefaultSetting(config, "ResNet.BatchNorm.Momentum", 0.99)
    self.BatchNormEpsilon       = DefaultSetting(config, "ResNet.BatchNorm.Epsilon", 0.001)
    self.DownSampling           = [False ] *len(self.StackSetup)
    self.DownSampling[1:]       = [True, True, True]
    self.DownSampling           = DefaultSetting(config, "ResNet.Blocks.IsDownsampling", self.DownSampling)
    self.IsLogitsOutput         = False
    self.IsGlobalAveragePooling = True
    self.IsClassifier           = True
    self.WeightDecay            = DefaultSetting(config, "Training.WeightDecay", 1e-4)
    self.ClassCount             = DefaultSetting(config, "Classcount", self.Features[-1])
    self.IsExplainableStem      = DefaultSetting(config, "ResNet.IsExplainableStem", False)

    # // Layers \\
    self.StemConv = None
    self.StemBN = None
    self.StemActivation = None
    self.ResidualBlocks = []
    self.LastBN = None
    self.LastActivation = None
    self.GlobalPool = None
    self.Logits = None
    self.Classes = None
    self.Explainer = None
    # ................................................................
    self.create_model()
  # --------------------------------------------------------------------------------------------------------------------
  def _get_stack_module_strides(self, stack_index):
    nModuleCount = self.StackSetup[stack_index]
    nStrides = [1] * nModuleCount
    if self.DownSampling[stack_index]:
      nStrides[0] = 2
    else:
      nStrides[1] = 1
    return nStrides
  # --------------------------------------------------------------------------------------------------------------------
  def create_model(self):
    # Stem
    print("Stem")
    print(" |_ Conv Layer Features:%d Stride:1" % self.StemFeatures[0])
    self.StemConv = Conv2D(self.StemFeatures[0], (3, 3), (1, 1), padding="same", use_bias=False,
                             kernel_initializer="he_uniform", kernel_regularizer=L2(self.WeightDecay))
    if self.IsExplainableStem:
      self.Explainer = Softmax(axis=3)
      print(" |_ Explainable Stem")

    print(f" |_ BatchNormalization momentum:{self.BatchNormMomentum} epsilon:{self.BatchNormEpsilon:.6f}")
    self.StemBN = BatchNormalization(momentum=self.BatchNormMomentum, epsilon=self.BatchNormEpsilon)
    print(f" |_ ReLU")
    self.StemActivation = Activation("relu")

    for nIndex in range(0, len(self.StackSetup)):
      print("Stack %d" % (nIndex +1))
      nStackModuleStrides = self._get_stack_module_strides(nIndex)
      for nModuleIndex, nModuleStride in enumerate(nStackModuleStrides):
        # bIsFirstModule = ((p_nStackIndex==0) and (nModuleIndex==0))
        oResBlock = ResNet.ResidualModuleClass(self.Features[nIndex], nModuleStride,
                                               batchnorm_momentum=self.BatchNormMomentum, batchnorm_epsilon=self.BatchNormEpsilon
                                               ,weight_decay=self.WeightDecay)
        self.ResidualBlocks.append(oResBlock)
        print(f" |_ ResBlock{nModuleIndex + 1} :{ResNet.ResidualModuleClass.__name__}  Features:{self.Features[nIndex]}  Stride:{nModuleStride}")

    print("Global Average Pooling")
    self.GlobalPool = GlobalAveragePooling2D()
    print(f"Dense {self.ClassCount} neurons")
    self.Logits = Dense(self.ClassCount, bias_initializer="zeros",
                        kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
    self.Classes = Softmax()
  # -------------------------------------------------------------------------------------------------------------------
  def call(self, input):
    tA = input

    tA = self.StemConv(tA)
    if self.IsExplainableStem:
      tLI = 1.0 + self.Explainer(tA)
      tA = tA * tLI
    else:
      tLI = tA

    tA = self.StemBN(tA)
    tA = self.StemActivation(tA)

    for nBlockIndex, oResBlock in enumerate(self.ResidualBlocks):
      tA = oResBlock(tA)

    tA = self.GlobalPool(tA)
    tA = self.Logits(tA)
    tY = self.Classes(tA)

    return tY
# =========================================================================================================================
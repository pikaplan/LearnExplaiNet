import tensorflow as tf
from keras import Model
from keras.layers import (
  Layer,
  Activation, Softmax,
  ZeroPadding2D,
  AveragePooling2D, GlobalAveragePooling2D,
  Conv2D, Dense,
  BatchNormalization
)

from keras.regularizers import L2

from .residual_module_constants import ResConnectionType, ResModuleType
from mllib.utils import DefaultSetting


# ...........................................................................
def resnet_infer_module_stack(layer_count):
  # Infer the block count from layers of ResNet
  nInferedBlockCount = None
  for nModulesPerBlock in [2, 3, 4, 5, 6]:
    for nBlockCount in [3, 4, 5]:
      if (2 * nModulesPerBlock * nBlockCount) == layer_count - 2:
        nInferedBlockCount = nBlockCount
        break
    if nInferedBlockCount is not None:
      break
  return nInferedBlockCount, nModulesPerBlock


# ...........................................................................
def resnet_tiny_infer_downsampling(infered_block_count):
  # Determine downsampling levels
  if infered_block_count > 5:
    raise Exception(f"Unsupported block count {infered_block_count}")
  oIsDownSampling = [True] * infered_block_count
  oIsDownSampling[0] = False
  if infered_block_count == 5:
    oIsDownSampling[2] = False
  return oIsDownSampling


# ...........................................................................

# =========================================================================================================================
class ResNetModule(Layer):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_nFeatures, p_nStrideOnInput, p_eModuleType, p_nWeightDecay, batchnorm_momentum=0.99,
               batchnorm_epsilon=0.001
               , p_nResidualConnectionType=ResConnectionType.AVG_POOL_PAD, p_bIsAnalyzing=False):
    super(ResNetModule, self).__init__()

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
      y = Activation("relu")(b + tResidual)
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





# =========================================================================================================================
class ResNet(Model):
  ResidualModuleClass = ResNetModule

  # --------------------------------------------------------------------------------------
  def __init__(self,
               p_oConfig):  # , p_nModuleCount=None, p_oConvFeatures=[], p_oWindowSizes=[], p_oPoolStrides=[], p_bHasBias=True, p_sActivationFunction="relu"):
    super(ResNet, self).__init__()
    # ................................................................
    # // Attributes \\
    # ................................................................
    self.ModuleType = p_oConfig["ResNet.ModuleType"]
    self.Features = p_oConfig["ResNet.Blocks.Features"]
    self.StackSetup = p_oConfig["ResNet.Blocks.Modules"]
    self.StemFeatures = DefaultSetting(p_oConfig, "ResNet.Stem.Features", [self.Features[0]])
    self.BatchNormMomentum = DefaultSetting(p_oConfig, "ResNet.BatchNorm.Momentum", 0.99)
    self.BatchNormEpsilon = DefaultSetting(p_oConfig, "ResNet.BatchNorm.Epsilon", 0.001)
    self.ResidualConnectionType = DefaultSetting(p_oConfig, "ResNet.ResidualConnectionType",
                                                 ResConnectionType.AVG_POOL_PAD)

    self.DownSampling = [False] * len(self.StackSetup)
    self.DownSampling[1:] = [True, True, True]
    self.DownSampling = DefaultSetting(p_oConfig, "ResNet.Blocks.IsDownsampling", self.DownSampling)
    self.IsLogitsOutput = False
    self.IsGlobalAveragePooling = True
    self.IsClassifier = True
    self.WeightDecay = DefaultSetting(p_oConfig, "Training.WeightDecay", 1e-4)
    print(f"WeightDecay:{self.WeightDecay:.5f}")
    # // Layers \\
    self.StemConv = None
    self.StemBN = None
    self.ResidualBlocks = []
    self.Logits = None
    self.LastBN = None
    self.GlobalPool = None
    self.Classes = None
    self.ClassCount = DefaultSetting(p_oConfig, "Classcount", self.Features[-1])
    self.LayerActivations = None
    self.IsAnalyzing = DefaultSetting(p_oConfig, "IsAnalyzing", False)
    # ................................................................
    self.CreateModel()

  # --------------------------------------------------------------------------------------------------------
  def ResidualModuleStack(self, p_nStackIndex, p_nFeatures, p_nModuleStrides):
    print("Stack %d" % (p_nStackIndex + 1))
    for nModuleIndex, nModuleInputStride in enumerate(p_nModuleStrides):
      # bIsFirstModule = ((p_nStackIndex==0) and (nModuleIndex==0))
      oResBlock = ResNet.ResidualModuleClass(p_nFeatures, nModuleInputStride, self.ModuleType,
                                             batchnorm_momentum=self.BatchNormMomentum,
                                             batchnorm_epsilon=self.BatchNormEpsilon
                                             , p_nResidualConnectionType=self.ResidualConnectionType
                                             , p_nWeightDecay=self.WeightDecay, p_bIsAnalyzing=self.IsAnalyzing)
      self.ResidualBlocks.append(oResBlock)
      print(" |_ ResBlock%d :%s  Features:%d  Stride:%d" % (
        nModuleIndex + 1, ResNet.ResidualModuleClass.__name__, p_nFeatures, nModuleInputStride))
  # --------------------------------------------------------------------------------------------------------
  def __getModuleStrides(self, p_nIndex):
    nModuleCount = self.StackSetup[p_nIndex]

    if self.DownSampling[p_nIndex]:
      nStrides = [2]
    else:
      nStrides = [1]
    nStrides = nStrides + [1] * (nModuleCount - 1)

    return nStrides
  # --------------------------------------------------------------------------------------
  def CreateModel(self):
    # https://github.com/D-X-Y/ResNeXt-DenseNet

    # Stem
    print("Stem")
    print(" |_ Conv Layer Features:%d Stride:1" % self.StemFeatures[0])
    self.StemConv = Conv2D(self.StemFeatures[0], (3, 3), (1, 1), padding="same", use_bias=False,
                           kernel_initializer="he_uniform"
                           , kernel_regularizer=L2(self.WeightDecay))

    if self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value:
      pass
    if self.ModuleType == ResModuleType.FULL_PREACTIVATION.value:
      pass
    else:
      print(f" |_ BatchNormalization momentum:{self.BatchNormMomentum} epsilon:{self.BatchNormEpsilon:.6f}")
      self.StemBN = BatchNormalization(momentum=self.BatchNormMomentum, epsilon=self.BatchNormEpsilon)

    for nIndex in range(0, len(self.StackSetup)):
      self.ResidualModuleStack(nIndex, self.Features[nIndex], self.__getModuleStrides(nIndex))

    if self.IsLogitsOutput:
      print("Logits Conv2D")
      self.Logits = Conv2D(self.Features[-1], (1, 1), (1, 1), padding="valid"
                           , kernel_regularizer=L2(self.WeightDecay))
    else:
      if (self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value) or (
              self.ModuleType == ResModuleType.FULL_PREACTIVATION.value):
        print(
          f"Representation BatchNormalization momentum:{self.BatchNormMomentum} epsilon:{self.BatchNormEpsilon:.6f}")
        self.LastBN = BatchNormalization(momentum=self.BatchNormMomentum, epsilon=self.BatchNormEpsilon)

    print("Global Average Pooling")
    if self.IsGlobalAveragePooling:
      self.GlobalPool = GlobalAveragePooling2D()

    if self.IsClassifier:
      self.Logits = Dense(self.ClassCount, bias_initializer="zeros", kernel_regularizer=L2(self.WeightDecay))
      self.Classes = Softmax()

  # --------------------------------------------------------------------------------------
  def call(self, input):
    tA = input
    tA = self.StemConv(tA)

    if self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value:
      pass
    if self.ModuleType == ResModuleType.FULL_PREACTIVATION.value:
      pass
    else:
      tA = self.StemBN(tA)
      tA = Activation("relu")(tA)

    for nBlockIndex, oResBlock in enumerate(self.ResidualBlocks):
      tA = oResBlock(tA)

    if self.IsLogitsOutput:
      tA = self.Logits(tA)
    else:
      if (self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION.value) or (
              self.ModuleType == ResModuleType.FULL_PREACTIVATION.value):
        tA = Activation("relu")(self.LastBN(tA))

    if self.IsGlobalAveragePooling:
      tA = self.GlobalPool(tA)
      tY = tA # Representation output

    if self.IsClassifier:
      tA = self.Logits(tA)
      tA = self.Classes(tA)
      tY = tA # Class confidence output

    return tA
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


from keras.layers import (
    Conv2D, Dense, BatchNormalization, Activation,
    Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
  )
from keras.regularizers import L2

from .residual_module_constants import  ResModuleType, ResConnectionType
from .ResNet import CResidualModule


# =======================================================================================================================
class CResidualBottleneckModule(CResidualModule):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_nFeatures, p_nStrideOnInput, p_nResidualType):
    self.Expansion = 4
    super(CResidualBottleneckModule, self).__init__(self, p_nFeatures, p_nStrideOnInput, p_nResidualType)

  # --------------------------------------------------------------------------------------
  def CreateBlock(self):
    self.SpatialDownSampling = self.CreateResidualConnection()
    self.ResidualBN = BatchNormalization()
    self.Pad1 = None
    self.Conv1 = Conv2D(self.Features, (1, 1), (1, 1), padding="valid", use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=L2(self.WeightDecay))
    self.BN1 = BatchNormalization()
    self.Pad2 = ZeroPadding2D((1, 1))
    self.Conv2 = Conv2D(self.Features, (3, 3), (self.StrideOnInput, self.StrideOnInput), padding="valid",
                        use_bias=False, kernel_initializer="he_uniform"
                        , kernel_regularizer=L2(self.WeightDecay))
    self.BN2 = BatchNormalization()
    self.Conv3 = Conv2D(self.Features * self.Expansion, (1, 1), (1, 1), padding="valid", use_bias=False,
                        kernel_initializer="he_uniform"
                        , kernel_regularizer=L2(self.WeightDecay))
    self.BN3 = BatchNormalization()
    # --------------------------------------------------------------------------------------

  def call(self, p_tInput):
    self.Input = p_tInput
    self.InputFeatures = self.Input.get_shape().as_list()[3]
    x = self.Input

    tResidual = self.CallResidualConnection(x)

    a = self.Conv1(x)
    a = self.BN1(a)
    a = Activation("relu")(a)

    a = self.Pad2(a)
    a = self.Conv2(a)
    a = self.BN2(a)
    a = Activation("relu")(a)

    a = self.Conv3(a)

    if self.ModuleType == ResModuleType.ORIGINAL.value:
      b = self.BN3(a)
      tResidualBN = self.ResidualBN(tResidual)
      y = Activation("relu")(b + tResidualBN)
    else:
      raise Exception("Not supported")

    self.Output = y
    return y
    # --------------------------------------------------------------------------------------------------------

# =======================================================================================================================

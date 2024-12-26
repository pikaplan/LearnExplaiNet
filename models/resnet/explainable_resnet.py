import numpy as np
import tensorflow as tf
import keras
import keras.layers as layers

import keras.backend as K
from mllib import RandomSeed
from mllib.utils import DefaultSetting

from models.resnet import ResModuleType
from explanations.ldp import LDPEncoder

from models.layer_activation_collection import CLayerActivationCollection
from models.layers import LateralInhibitionExplainer
from models.layers import SpatialWindowSlicer
from models.layers.layer_spatial_dims import CNNSpatialDims, LayerSpatialDims



# =========================================================================================================================
class LDFExResidualModule(layers.Layer):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_nFeatures, p_nStrideOnInput, p_nStrideOnExplanations
                  , p_eModuleType, p_nWeightDecay, p_bIsAnalyzing=False, p_nVerboseLevel=1):
        super(LDFExResidualModule, self).__init__()

        # ................................................................
        # // Attributes \\
        self.Features             = p_nFeatures
        self.StrideOnInput        = p_nStrideOnInput
        self.StrideOnExplanations = p_nStrideOnExplanations
        self.ModuleType           = p_eModuleType        
        
        self.Expansion            = 1 
        self.WeightDecay          = p_nWeightDecay
        
        
        self.MustCheckToPadFeatures = True
        self.IsAnalyzing      = p_bIsAnalyzing
        self.VerboseLevel     = p_nVerboseLevel
        self.LayerActivations = None      

        
        # // Layers \\
        self.SpatialDownSampling = None
        self.ResidualBN       = None
        self.Pad1             = None
        self.Conv1            = None
        self.BN1              = None
        self.Activ1           = None
        self.Pad2             = None
        self.Conv2            = None
        self.BN2              = None
        self.Activ2           = None
        self.LateralInhibitor = None
        self.ExplanationSlicer  = None
        self.ExplanationEncoder = None
            
        self.Input            = None
        self.InputFeatures    = None
        self.Output           = None
        
        self.ExplanationModule = None
        # ................................................................
        self.CreateBlock()
    # --------------------------------------------------------------------------------------------------------
    def CreateResidualConnection(self):
        self.MustCheckToPadFeatures = True
        
        tSpatialDownsampling = None
        if True:
            # Downsample spatially
            if (self.StrideOnInput > 1):
                tSpatialDownsampling = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
                if self.VerboseLevel >= 1:
                  print("     |__ Average pooling on residual connection input")
        else:
            if (self.StrideOnInput > 1):
                if True:
                    nWindowSize = (self.ResidualType, self.ResidualType)
                    nConvStride = (self.StrideOnInput, self.StrideOnInput)
                    tSpatialDownsampling = layers.Conv2D( self.Features, kernel_size=nWindowSize, strides=nConvStride
                                                   ,padding="same",use_bias=False, kernel_initializer="he_uniform"
                                                   ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
                    if self.VerboseLevel >= 1:
                      print("     |__ Conv2D on residual connection input")
                self.MustCheckToPadFeatures = False
        if tSpatialDownsampling is None:
          if self.VerboseLevel >= 1:
            print("     |__ no adaptation for residual connection input")
        return tSpatialDownsampling
    # --------------------------------------------------------------------------------------------------------
    def PadFeatures(self, p_tX, p_nFeatureDepth, p_sName=None):
        nInputShape = p_tX.shape.as_list()

        nPadFeatures = p_nFeatureDepth - nInputShape[3]
        tPad =  tf.pad(p_tX, [[0, 0], [0, 0], [0, 0], [nPadFeatures // 2, nPadFeatures // 2]], name=p_sName)
        
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
                
        return tResidual
    # --------------------------------------------------------------------------------------     
    def TransferParamsFrom(self, p_oParamsDict):
        if "SpatialDownSampling" in p_oParamsDict:
            nWeights = p_oParamsDict["SpatialDownSampling"]
            if nWeights != []:
                self.SpatialDownSampling.set_weights(nWeights)
        if "nWeights" in p_oParamsDict:
            nWeights = p_oParamsDict["ResidualBN"]
            if nWeights != []:
                self.ResidualBN.set_weights(nWeights)
            
        self.Conv1.set_weights(p_oParamsDict["Conv1"])
        self.BN1.set_weights(p_oParamsDict["BN1"])
        self.Conv2.set_weights(p_oParamsDict["Conv2"])
        self.BN2.set_weights(p_oParamsDict["BN2"])
      
    # --------------------------------------------------------------------------------------
    def CreateBlock(self):
        self.SpatialDownSampling = self.CreateResidualConnection()
        self.ResidualBN   = layers.BatchNormalization()
        self.Pad1        = layers.ZeroPadding2D((1,1))
        
        self.Conv1       = layers.Conv2D(self.Features, (3,3), (self.StrideOnInput, self.StrideOnInput)
                                         , padding="valid", use_bias=False, kernel_initializer="he_uniform"
                                         ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
        self.BN1         = layers.BatchNormalization()
        self.Activ1      = layers.Activation("relu")
        self.Pad2        = layers.ZeroPadding2D((1,1))
        self.Conv2       = layers.Conv2D(self.Features, (3,3), (1,1), padding="valid"
                                  , use_bias=False, kernel_initializer="he_uniform"
                                  ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
        self.LateralInhibitor   = LateralInhibitionExplainer()
        #self.LateralInhibitor   = LateralInhibitionPyramidExplainer()
        
        self.BN2         = layers.BatchNormalization()
        self.Activ2      = layers.Activation("relu")    
        
        
        #self.ExplanationSlicer  = SpatialWindowSlicer((3,3), (self.StrideOnExplanations, self.StrideOnExplanations))
        #self.ExplanationEncoder = LDPEncoder(self.Features, 3)
    # --------------------------------------------------------------------------------------
    def recordActivationStart(self):
      if self.IsAnalyzing:
        if self.LayerActivations is None:
          self.LayerActivations = CLayerActivationCollection()
        self.LayerActivations.NewSample()       
    # --------------------------------------------------------------------------------------
    def recordActivation(self, p_tTensor):
        if self.IsAnalyzing:
            self.LayerActivations.append(p_tTensor.numpy())
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        self.Input = p_tInput
        self.recordActivationStart()
        
        self.InputFeatures  = self.Input.get_shape().as_list()[3]
        tX = self.Input        
        
        tResidual = self.CallResidualConnection(tX)
       
        # ...... Convolutional module 1 ......
        tA = self.Pad1(tX)
        tA = self.Conv1(tA)
        tA = self.BN1(tA)
        tA = self.Activ1(tA)

        # ...... Convolutional module 2 (with Lateral Inhibition + Explainer) ......
        tA = self.Pad2(tA)    
        tA = self.Conv2(tA)
        tG, tSoftmax, tExplanation = self.LateralInhibitor(tA)
        self.recordActivation(tG)
        
        tB = self.BN2(tG)
        tY = self.Activ2(tB + tResidual)
        
        
        tLDFs     = tf.argsort(tSoftmax, axis=3, direction='DESCENDING', stable=False)
        tLDFProbs = tf.sort(tSoftmax, axis=3, direction='DESCENDING')
        
        #tLDPs = self.ExplanationSlicer(tExplanation)
        #tLDPCodes, tLDPFeatureHist, _, _ = self.ExplanationEncoder(tLDPs)
  
        self.Output = tY   
        
        return tY, tSoftmax, tExplanation, tLDFs, tLDFProbs, None
    # --------------------------------------------------------------------------------------        
# =========================================================================================================================
  




# =========================================================================================================================
class ExplainableResNet(keras.Model):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfig, resmodule_class=LDFExResidualModule, p_nVerboseLevel=1):#, p_nModuleCount=None, p_oConvFeatures=[], p_oWindowSizes=[], p_oPoolStrides=[], p_bHasBias=True, p_sActivationFunction="relu"):
        super(ExplainableResNet, self).__init__()
        # ................................................................
        # // Attributes \\
        # ................................................................
        self.ResModuleClass   = resmodule_class
        self.VerboseLevel     = p_nVerboseLevel
        self.ModuleType       = p_oConfig["ResNet.ModuleType"]
        self.Features         = p_oConfig["ResNet.Blocks.Features"]
        self.StackSetup       = p_oConfig["ResNet.Blocks.Modules"]
        self.DownSampling     = [False]*len(self.StackSetup) 
        self.DownSampling[1:] = [True, True, True]
        self.DownSampling     = DefaultSetting(p_oConfig, "ResNet.IsDownsampling", self.DownSampling)
        
        self.ModuleStrides    = [1]*np.sum(self.StackSetup)
        nCurrentModule = 0
        for nStackIndex, bIsDownSampling in enumerate(self.DownSampling):
          if bIsDownSampling:
            self.ModuleStrides[nCurrentModule] = 2
          nCurrentModule += self.StackSetup[nStackIndex]
        
        self.IsLogitsOutput         = False
        self.IsGlobalAveragePooling = True
        self.IsClassifier   = True
        self.WeightDecay    = DefaultSetting(p_oConfig, "Training.WeightDecay", 1e-4)
            
        # // Layers \\
        self.StemConv = None
        self.StemLateralInhibitor = None
        self.StemBN = None
        self.StemActivation = None
        self.ResidualBlocks = []
        self.Logits = None
        self.LastBN = None
        self.LastActivation = None
        self.GlobalPool = None
        self.ClassProbs = None
        
        self.ClassCount = DefaultSetting(p_oConfig, "ClassCount", self.Features[-1])
        self.LayerActivations = None  
        self.IsAnalyzing = DefaultSetting(p_oConfig, "IsAnalyzing", False)      
        self.IsExplainableStem = DefaultSetting(p_oConfig, "IsExplainableStem", False)
        self.Config = p_oConfig
        
        self.ExplanationModel = None
        
        self.__moduleIndex = 0
        self.layer_spatial_dims = CNNSpatialDims()
        # ................................................................
        self.CreateModel()
    # --------------------------------------------------------------------------------------------------------
    def ResidualModuleStack(self, p_nStackIndex, p_nFeatures, p_nModuleStrides):
        if self.VerboseLevel >= 1:
          print("Stack %d" % (p_nStackIndex+1))
        
        for nModuleInputStride in p_nModuleStrides:
            nStrideOnExplanations = 1 
            if self.__moduleIndex < (len(self.ModuleStrides) - 1):
              nStrideOnExplanations = self.ModuleStrides[self.__moduleIndex + 1]
            #print("Explanations: %s.%d" % (p_nStackIndex, nModuleIndex), p_nFeatures, nModuleInputStride, nStrideOnExplanations)
            oResBlock = self.ResModuleClass(p_nFeatures, nModuleInputStride, nStrideOnExplanations
                                                    ,self.ModuleType, p_nWeightDecay=self.WeightDecay
                                                    , p_bIsAnalyzing=self.IsAnalyzing
                                                    , p_nVerboseLevel=self.VerboseLevel)
            self.ResidualBlocks.append(oResBlock)
            oResBlock.ExplanationModel = self.ExplanationModel
            if self.VerboseLevel >= 1:
              print(" |_ ResBlock%d :%s  Features:%d  Stride:%d  StrideOnExplanations:%d"  % (self.__moduleIndex+1, 
                      "CExplainableResidualModule", p_nFeatures, nModuleInputStride, nStrideOnExplanations))
            self.__moduleIndex += 1
    # --------------------------------------------------------------------------------------------------------
    def __getModuleStrides(self, p_nIndex):
        nModuleCount = self.StackSetup[p_nIndex]
    
        if  self.DownSampling[p_nIndex]: 
            nStrides = [2]
        else:
            nStrides = [1]
        nStrides = nStrides + [1]*(nModuleCount -1)
        
        return nStrides   
    # --------------------------------------------------------------------------------------------------------
    def TransferParamsFrom(self, p_oParamsDict):
        self.StemConv.set_weights(p_oParamsDict["conv2d"])
        self.StemBN.set_weights(p_oParamsDict["batch_normalization"])
        #self.LastBN.set_weights(p_oParamsDict["LastBN"])
        self.Logits.set_weights(p_oParamsDict["dense"])
        
        for nIndex,oBlock in enumerate(self.ResidualBlocks):
            # TODO: Ordering independence
            sName = "c_ex_residual_module"
            if nIndex > 0:
                sName += "_%d" % nIndex
            oBlock.TransferParamsFrom(p_oParamsDict[sName]) 
    # --------------------------------------------------------------------------------------
    def CreateModel(self):
        # Stem
        print("Stem")
        self.StemConv = layers.Conv2D(self.Features[0], (3,3), (1,1), padding="same", use_bias=False, kernel_initializer="he_uniform"
                               ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
        
        if self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION:
            pass
        if self.ModuleType == ResModuleType.FULL_PREACTIVATION:
            pass
        else:
            print(" |_ Stem BatchNormalization")
            self.StemBN = layers.BatchNormalization()

        self.StemActivation = layers.Activation("relu")
        
        self.__moduleIndex = 0
        for nIndex in range(0, len(self.StackSetup)):
            self.ResidualModuleStack(nIndex, self.Features[nIndex], self.__getModuleStrides(nIndex))
      
        if self.IsLogitsOutput:
            print("Logits Conv2D")
            self.Logits = layers.Conv2D(self.Features[-1], (1,1), (1,1), padding="valid"
                                 ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
        else:
            if (self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION) or (self.ModuleType == ResModuleType.FULL_PREACTIVATION):
                self.LastActivation = layers.Activation("relu")
                print("Representation BatchNormalization")
                self.LastBN = layers.BatchNormalization()
                
        
        print("Global Average Pooling")
        if self.IsGlobalAveragePooling:
            self.GlobalPool = layers.GlobalAveragePooling2D()
        
        if self.IsClassifier:
            self.Logits   = layers.Dense(self.ClassCount, bias_initializer="zeros", kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))    
            self.ClassProbs  = layers.Softmax()
                  
    # --------------------------------------------------------------------------------------
    def recordActivationStart(self):
      if self.IsAnalyzing:
        if self.LayerActivations is None:
          self.LayerActivations = CLayerActivationCollection()
        self.LayerActivations.NewSample()   
    # --------------------------------------------------------------------------------------
    def recordActivation(self, p_oTensorOrModule, p_bIsModule=False, p_bIsNewSample=False):
        if self.IsAnalyzing:
            if not p_bIsModule:
                self.LayerActivations.append(p_oTensorOrModule.numpy())
            else:
                self.LayerActivations.appendModule(p_oTensorOrModule.LayerActivations)
    # --------------------------------------------------------------------------------------
    def build(self, input_shape):
        nShape = list(input_shape)
        oInputDims = LayerSpatialDims(None, input_dim=nShape[1], window_size=1,
                                      layer_index=len(self.layer_spatial_dims))
        self.layer_spatial_dims.append(oInputDims)
        oStemDims = LayerSpatialDims(oInputDims, window_size=3, stride=1,
                                     layer_index=len(self.layer_spatial_dims))
        self.layer_spatial_dims.append(oStemDims)
        for oExplainableResBlock in self.ResidualBlocks:
          for nLayerIndex in range(2):
            if nLayerIndex == 0:
              nStride = oExplainableResBlock.StrideOnInput
            else:
              nStride = 1
            oLayerDims = LayerSpatialDims(self.layer_spatial_dims[-1], 
                                          window_size=3,stride=nStride,
                                          layer_index=len(self.layer_spatial_dims))
            self.layer_spatial_dims.append(oLayerDims)
            
                    
        super(ExplainableResNet, self).build(input_shape)
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        if isinstance(p_tInput, tuple):
          tIDs, tA = p_tInput
        else:
          tIDs, tA = None, p_tInput
          
        if self.ExplanationModel is not None:
          self.ExplanationModel.start_initialization()
        
        self.recordActivationStart()
        tA = self.StemConv(tA)
        if self.IsExplainableStem:
          tA, _, tExplanation = self.StemLateralInhibitor(tA)
          self.recordActivation(tExplanation)

        if self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION:
            pass
        if self.ModuleType == ResModuleType.FULL_PREACTIVATION:
            pass
        else:
            tA = self.StemBN(tA)
            tA = self.StemActivation(tA)
        
        for nIndex, oExplainableResBlock in enumerate(self.ResidualBlocks):
            tA, tSoftmax, tExplanation, tLDPs, tLDPCodes, tLDPFeatureHist = oExplainableResBlock(tA)
            self.recordActivation(oExplainableResBlock, p_bIsModule=True)
            if self.ExplanationModel is not None:
                oExModule = self.ExplanationModel\
                              .next_module(True, oExplainableResBlock.Features)\
                              .link_tensors(tIDs, tSoftmax, tExplanation, tLDPs, tLDPCodes, tLDPFeatureHist)

                if oExModule is not None:
                    oExplainableResBlock.ExplanationModule = oExModule
        
        
        if self.IsLogitsOutput:
            tA = self.Logits(tA)   
        else:
            if (self.ModuleType == ResModuleType.RELU_ONLY_PREACTIVATION) or (self.ModuleType == ResModuleType.FULL_PREACTIVATION):
                tA = self.LastActivation(self.LastBN(tA))

        if self.IsGlobalAveragePooling:
            tA = self.GlobalPool(tA)
            self.recordActivation(tA)
            if self.ExplanationModel is not None:
                self.ExplanationModel.next_module(False).link_tensors(tIDs, tA)
            
        if self.IsClassifier:
            tA = self.Logits(tA)
            self.recordActivation(tA)
            y = self.ClassProbs(tA)
            tClasses = K.expand_dims(K.argmax(y), 1)
            if self.ExplanationModel is not None:
                self.ExplanationModel.next_module(False).link_tensors(tIDs, tClasses)
                
        if self.ExplanationModel is not None:
            self.ExplanationModel.end_initialization()
        
        return y, tClasses, tIDs
    # --------------------------------------------------------------------------------------

# =========================================================================================================================

  
def __testExModule():
  from mllib import PrintTensor
  VERBOSE = False
  FEATURE_COUNT = 1024
  SAMPLE_COUNT  = 128
  GRID_SIZE     = 55
  
  RandomSeed(2023)    
  nInput = np.random.rand(SAMPLE_COUNT,GRID_SIZE,GRID_SIZE,FEATURE_COUNT).astype(np.float32)
  if VERBOSE:
    PrintTensor("Input Data", nInput)
  
  oExplainer = LateralInhibitionExplainer()
  oSlicer    = SpatialWindowSlicer((3,3), (1,1))
  oEncoder   = LDPEncoder(FEATURE_COUNT, oSlicer.WindowSize[0])
  #oMultihot  = keras.layers.CategoryEncoding(num_tokens=2**oEncoder.PixelCount, output_mode="multi_hot")

  
  oSlicer.run_eagerly = True
  oEncoder.run_eagerly = True
  
  tActivation, tProbs, tExplanation = oExplainer(nInput)
  tSlices = oSlicer(tExplanation)
  tCodes, tHist, tCodesReshaped, tHistReshaped = oEncoder(tSlices)
  

  
  nActivation   = tActivation.numpy()
  nExplanation  = tExplanation.numpy()
  nSlices       = tSlices.numpy()
  nCodes        = tCodes.numpy()
  nHist         = tHist.numpy()
      
  
  if VERBOSE:
    PrintTensor("Activation", nActivation)
    PrintTensor("Explanation", nExplanation)
    print("Explanation Slices %s" % str(nSlices.shape))
    for nSampleIndex in range(5):
      PrintTensor("Sample #%d" % nSampleIndex, nSlices)
    

    print("LDP shapes:", nCodes.shape, nHist.shape)  
    print("Unique Pattern Code Count:%d" % len(np.unique(nCodes)))
  
    PrintTensor("LDP Slices", nCodes)
  else:
    print("Activation", nActivation.shape)
    print("Explanation", nExplanation.shape)
    print("Explanation Module Input", nSlices.shape)
    print("LDP Patterns", nCodes.shape)
    print("LDP Pattern Feature Bins", nHist.shape)
    
    nPC = len(np.unique(nCodes))*100.0/512.0
    print("Unique Pattern Code Count %.1f%% out of total" % nPC)
    print("Unique LDPs:%d" % len(np.unique(nCodes.reshape([np.prod(nCodes.shape[0:3]), nCodes.shape[-1]]), axis=1)))
      
      
    
if __name__ == "__main__":
  __testExModule()
    
          
    
  

    
  


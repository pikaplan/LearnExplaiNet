import tensorflow as tf
import keras.layers as layers
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.layers import InputLayer, Flatten, Dense, BatchNormalization, Activation, Softmax, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.regularizers import L2

from models.resnet.residual_module_constants import ResConnectionType

from models.layer_activation_collection import CLayerActivationCollection

# =========================================================================================================================
class CExResidualModule(layers.Layer):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_nFeatures, p_nStrideOnInput, p_eModuleType, p_nWeightDecay, batchnorm_momentum=0.99,
               batchnorm_epsilon=0.001, p_nResidualConnectionType=ResConnectionType.AVG_POOL_PAD, p_bIsAnalyzing=False):
        super(CExResidualModule, self).__init__()

        # ................................................................
        # // Attributes \\
        self.Features       = p_nFeatures
        self.StrideOnInput  = p_nStrideOnInput
        self.ModuleType             = p_eModuleType        
        #self.ResidualConnectionType = p_nResidualConnectionType
        self.Expansion      = 1 
        self.WeightDecay    = p_nWeightDecay
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_epsilon = batchnorm_epsilon
        #self.DropOutRate    = 0.5
        self.BlockType = 2
        self.MustCheckToPadFeatures = True
        
        # // Layers \\
        self.Input          = None
        self.InputFeatures  = None
        self.SpatialDownSampling = None
        self.Conv1 = None
        self.Conv2 = None
        self.Conv3 = None
        self.Activation = None
        self.Softmax = None
        self.Output = None
        
        self.IsAnalyzing = p_bIsAnalyzing
        self.LayerActivations = None      
        # ................................................................
        self.CreateBlock()
    # --------------------------------------------------------------------------------------------------------
    def CreateResidualConnection(self):
        self.MustCheckToPadFeatures = True
        
        tSpatialDownsampling = None
        if True:
            # Downsample spatially
            if (self.StrideOnInput > 1):
                tSpatialDownsampling = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
                print("     |__ Average pooling on residual connection input")
        else:
            if (self.StrideOnInput > 1):
                if True:
                    nWindowSize = (self.ResidualType, self.ResidualType)
                    nConvStride = (self.StrideOnInput, self.StrideOnInput)
                    tSpatialDownsampling = Conv2D( self.Features, kernel_size=nWindowSize, strides=nConvStride
                                                   ,padding="same",use_bias=False, kernel_initializer="he_uniform"
                                                   ,kernel_regularizer=tf.keras.regularizers.L2(self.WeightDecay))
                    print("     |__ Conv2D on residual connection input")
                self.MustCheckToPadFeatures = False
        if tSpatialDownsampling is None:
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
        else:
            if self.MustCheckToPadFeatures and (self.InputFeatures != self.Features):
                tResidual = self.PadFeatures(tResidual, self.Features, p_sName="pad_skip")
                                
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
        #//TODO: Inherited
        self.SpatialDownSampling = self.CreateResidualConnection()
        self.ResidualBN = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
        self.Pad1 = ZeroPadding2D((1, 1))
        self.Conv1 = Conv2D(self.Features, (3, 3), (self.StrideOnInput, self.StrideOnInput), padding="valid",
                            use_bias=False, kernel_initializer="he_uniform"
                            , kernel_regularizer=L2(self.WeightDecay))
        self.BN1 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)
        self.Pad2 = ZeroPadding2D((1, 1))
        self.Conv2 = Conv2D(self.Features, (3, 3), (1, 1), padding="valid", use_bias=False,
                            kernel_initializer="he_uniform"
                            , kernel_regularizer=L2(self.WeightDecay))
        self.BN2 = BatchNormalization(momentum=self.batchnorm_momentum, epsilon=self.batchnorm_epsilon)

        self.Softmax     = Softmax(axis=3)
    # --------------------------------------------------------------------------------------
    def recordActivation(self, p_tTensor, p_bIsNewSample=False):
        if self.IsAnalyzing:
            if p_bIsNewSample:
              if self.LayerActivations is None:
                self.LayerActivations = CLayerActivationCollection()
              self.LayerActivations.NewSample()
                        
            self.LayerActivations.append(p_tTensor.numpy())        
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        self.Input = p_tInput
        self.InputFeatures  = self.Input.get_shape().as_list()[3]
        x = self.Input        
        
        tResidual = self.CallResidualConnection(x)
       
       
        a = self.Pad1(x)
        a = self.Conv1(a)
        a = self.BN1(a)
        a = Activation("relu")(a)

        a = self.Pad2(a)    
        a = self.Conv2(a)
        #tSoftmaxInput = g * a # g = gain,  Scheduled   # <---- [Model 5]
        tSoftmaxInput = a
        if self.BlockType == 2:
            tY = 1.0 + self.Softmax(tSoftmaxInput)
            a = a * tY
            #a = a + a*tY  #Model 3   <-- We need to add a so that gradient will still flow for the disabled activations done by the softmax layer
        else:
            tY = self.Softmax(a) 
            a = a*tY   #Model 4
        self.recordActivation(tY, True)
        
        b = self.BN2(a)
        y = Activation("relu")(b + tResidual)
        

        self.Output = y   
        
        return self.Output
    # --------------------------------------------------------------------------------------
    ''''
    def call_default(self, p_tInput):
        self.Input = p_tInput
        self.InputFeatures  = self.Input.get_shape().as_list()[3]
        x = self.Input        
        
        tResidual = self.CallResidualConnection(x)
        
       
        a = self.Pad1(x)
        a = self.Conv1(a)
        a = self.BN1(a)
        a = Activation("relu")(a)

        a = self.Pad2(a)    
        a = self.Conv2(a)
        #tSoftmaxInput = g * a # g = gain,  Scheduled   # <---- [Model 5]
        tSoftmaxInput = a
        if self.BlockType == 2:
            tY = self.Softmax(tSoftmaxInput)
            a = a + a*tY  #Model 3   <-- We need to add a so that gradient will still flow for the disabled activations done by the softmax layer
        else:
            tY = self.Softmax(a) 
            a = a*tY   #Model 4
        b = self.BN2(a)
        y = Activation("relu")(b + tResidual)


        self.Output = y   
        
        return self.Output    
    '''
    # --------------------------------------------------------------------------------------        
# =========================================================================================================================










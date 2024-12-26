import keras
from keras import layers
from keras import backend as K

# =========================================================================================================================
class LateralInhibitionExplainer(keras.Model):
  # -----------------------------------------------------------------------------
  def __init__(self):
    super(LateralInhibitionExplainer, self).__init__()
    self.Softmax = layers.Softmax(axis=3)
  # -------------------------------------------------------------------------------------- 
  def call(self, p_nInput):
    tA = p_nInput
    tS = self.Softmax(tA)
    # ____________________________________________
    #   g(x) = a(x)*softmax(a(x)) + a(x) 
    #
    #   Activations of neurons a(x) are inhibited/excited by the softmax(a(x)) confidence scores
    # ____________________________________________
    tG = tA * (1.0 + tS)  
    tExplanation = K.expand_dims(K.argmax(tS), -1) + 1
    
    return tG, tS, tExplanation
  # --------------------------------------------------------------------------------------
# ========================================================================================= 


  
  
  
  
  
  
   
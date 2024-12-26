# ......................................................................................
# MIT License

# Copyright (c) 2024 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................

import numpy as np
from .DatasetBase import CDataSetBase


# --------------------------------------------------------------------------------------
def SequenceClips(p_nSamples, p_nLabels, p_nWindowSize, p_nStride, p_bIsPaddingZeros=False):
  nSequenceIndex = 0
  while nSequenceIndex < p_nSamples.shape[0]:
    nLabel    = p_nLabels[nSequenceIndex]
    nPosition = 0
    nSpanPoints = p_nWindowSize
    if p_bIsPaddingZeros:
      nSpanPoints = p_nWindowSize - 3*p_nStride
    
    nDataPointCount = p_nSamples.shape[1] 
    while (nPosition + nSpanPoints) <= nDataPointCount:
      if p_bIsPaddingZeros and ((nPosition + p_nWindowSize) >= nDataPointCount):
        nSeqSample = np.zeros((p_nWindowSize, p_nSamples.shape[2]), np.float32)
        nSeqSample[nPosition + p_nWindowSize - nDataPointCount:,:] = p_nSamples[nSequenceIndex, nPosition:, :]
      else:
        nSeqSample = p_nSamples[nSequenceIndex, nPosition:nPosition + p_nWindowSize, :]
      
      yield (nSeqSample, nLabel)
      
      nPosition += p_nStride
    nSequenceIndex += 1
# --------------------------------------------------------------------------------------

      
# =========================================================================================================================
class CSequenceDatasetBase(CDataSetBase):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_sName=None, p_nRandomSeed=None):
    super(CSequenceDatasetBase, self).__init__(p_sName=p_sName, p_nRandomSeed=p_nRandomSeed)
    # ................................................................
    # // Fields \\
    self.ClipWindowSize = None
    self.ClipStride     = None
    self.IsPaddingZeros = False
  # --------------------------------------------------------------------------------------
  @property
  def TSSequenceClips(self):
    return SequenceClips(self.TSSamples, self.TSLabels
                          , self.ClipWindowSize, self.ClipStride, self.IsPaddingZeros) 
  # --------------------------------------------------------------------------------------
  @property
  def VSSequenceClips(self):
    return SequenceClips(self.VSSamples, self.VSLabels
                          , self.ClipWindowSize, self.ClipStride, self.IsPaddingZeros) 
  # --------------------------------------------------------------------------------------
  def ConvertSamplesToClips(self, p_nWindowSize, p_nStride, p_bIsPaddingZeros=False):
    self.ClipWindowSize = p_nWindowSize
    self.ClipStride     = p_nStride
    self.IsPaddingZeros = p_bIsPaddingZeros
    
    nClips = [] 
    nClipLabels = []
    for (nClip, nClipLabel) in self.TSSequenceClips:
      nClips.append(nClip)
      nClipLabels.append(nClipLabel)      
    nClips = np.asarray(nClips)
    nClipLabels = np.asarray(nClipLabels)
  
    self.TSSamples = nClips 
    self.TSLabels = nClipLabels
  
    nClips = [] 
    nClipLabels = []
    for (nClip, nClipLabel) in self.VSSequenceClips:
      nClips.append(nClip)
      nClipLabels.append(nClipLabel)
    nClips = np.asarray(nClips)
    nClipLabels = np.asarray(nClipLabels)
    
    self.VSSamples = nClips
    self.VSLabels = nClipLabels
    
    self.countSamples() 
  # --------------------------------------------------------------------------------------
  
# =========================================================================================================================        
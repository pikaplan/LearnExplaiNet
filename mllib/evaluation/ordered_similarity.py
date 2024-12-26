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

# --------------------------------------------------------------------------------------
def ordered_set_similarity(true_set, predicted_set):
  nItemCount = len(true_set)
  assert len(predicted_set) == nItemCount, f"Invalid set size. Should be {nItemCount}"

  #weights   = 1.0 / np.log2( np.arange(nItemCount) + 2 ) 
  weights     =  2.0**(np.arange(nItemCount)[::-1])
  nEqual        = np.array(true_set[:] == predicted_set[:]).astype(np.float32)
  nShiftedEqual = np.array(true_set[:3] == predicted_set[1:]).astype(np.float32)*(1/2)
  nEqual[1:] += nShiftedEqual
 
  nNorm       = np.dot(weights, np.ones((nItemCount), np.float32))
  nVal        = np.dot(weights, nEqual)
  #nIOU        = iou(true_set, predicted_set)
  nResult     = (nVal/nNorm)# + nIOU/8.0
   
  return nResult
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
  oList  = np.array([8, 1, 4, 0])
  oList0 = np.array([8, 0, 1, 5])
  oList1 = np.array([8, 1, 4, 5])
  oList2 = np.array([8, 1, 2, 0])
  oList3 = np.array([1, 8, 4, 0])  
  oList4 = np.array([1, 8, 4, 5])

  print(ordered_set_similarity(oList, oList))
  print(ordered_set_similarity(oList, oList0))  
  print(ordered_set_similarity(oList, oList1))
  print(ordered_set_similarity(oList, oList2))
  print(ordered_set_similarity(oList, oList3))  
  print(ordered_set_similarity(oList, oList4))
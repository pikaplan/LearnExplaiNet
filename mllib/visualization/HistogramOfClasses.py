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

import matplotlib.pyplot as plt


# =========================================================================================================================
class CHistogramOfClasses(object):  # class CPlot: object
  # --------------------------------------------------------------------------------------
  def __init__(self, p_nData, p_nClasses, p_bIsProbabilities=False):
    self.Data = p_nData
    self.Classes = p_nClasses
    self.IsProbabilities = p_bIsProbabilities
  # --------------------------------------------------------------------------------------
  def Show(self):

    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.hist(self.Data, density=self.IsProbabilities, bins=self.Classes, ec="k") 
    ax.locator_params(axis='x', integer=True)

    if self.IsProbabilities:
      plt.ylabel('Probabilities')
    else:
      plt.ylabel('Counts')
    plt.xlabel('Classes')
    plt.show()
  # --------------------------------------------------------------------------------------


# =========================================================================================================================
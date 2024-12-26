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
from matplotlib import colors

class CClassColorMap(object):
  def __init__(self, p_sColorNames=["black", "red", "green", "blue", "yellow"]):
    self.ColorNames = p_sColorNames
    self.ColorCount = len(self.ColorNames)
  
 
  def Make(self):
    self.ColorCount = len(self.ColorNames)
    
    # Map a class number to a color
    nRange = range(self.ColorCount)
    oDict = {   nIndex: colors.to_rgb(self.ColorNames[nIndex]) for nIndex in nRange }
    
    # Create a colormap (optional)
    oColorRGB = [oDict[nIndex] for nIndex in nRange]
    oColorMap = colors.ListedColormap(oColorRGB)
    #oMyColorMapNorm = colors.BoundaryNorm(np.arange(C_p + 1) - 0.5, C_p)
    oColorMapNorm = colors.BoundaryNorm(np.asarray(range(self.ColorCount + 1), np.float32)
                                        , self.ColorCount)
    return oColorMap, oColorMapNorm
    
  
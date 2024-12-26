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

import csv


# =========================================================================================================================
class CModelConfig(object):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oModel, p_oValuesDictionary):
        self.Model = p_oModel
        self.Value = dict()
        for sKey in p_oValuesDictionary.keys():
            self.Value[sKey] = p_oValuesDictionary[sKey]
    # --------------------------------------------------------------------------------------
# =========================================================================================================================

        


# =========================================================================================================================
class CKerasModelStructureElement(list):
    # --------------------------------------------------------------------------------------    
    # Constructor
    def __init__(self, p_sName, p_oShape):
        # ..................... Object Attributes ...........................
        self.Name  = p_sName
        self.Shape = p_oShape
        # ...................................................................
    # --------------------------------------------------------------------------------------
    def __str__(self):
        return "%64s %s" % (self.Name, self.Shape)
        
    # --------------------------------------------------------------------------------------
# =========================================================================================================================

# =========================================================================================================================
class CKerasModelStructure(list):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    # ..................... Object Attributes ...........................
    self.SoftmaxActivation  = None
    self.LayerNumber = 0
    # ...................................................................
  # --------------------------------------------------------------------------------------
  def Add(self, p_tTensor):
      self.append(CKerasModelStructureElement(p_tTensor.name, p_tTensor.shape))
  # --------------------------------------------------------------------------------------
  def Print(self, p_sWriteToFileName=None):
      if p_sWriteToFileName is None:
            for nIndex,oElement in enumerate(self):
                print(nIndex+1, oElement.Name, oElement.Shape) 
      else:
        with open(p_sWriteToFileName, 'w') as f: 
            write = csv.writer(f)
            for nIndex,oElement in enumerate(self):
                print(nIndex, oElement) 
                write.writerow("%d;%s;%s" % (nIndex+1, oElement.Name, oElement.Shape))
  # --------------------------------------------------------------------------------------        
# =========================================================================================================================
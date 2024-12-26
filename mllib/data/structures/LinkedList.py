# ......................................................................................
# MIT License

# Copyright (c) 2020-2024 Pantelis I. Kaplanoglou

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

# =========================================================================================
class LinkedListItem(list):
  # -----------------------------------------------------------------------------
  def __init__(self, p_oValue=None):
    self.value  = p_oValue
    self.next   = None    
  # -----------------------------------------------------------------------------
  def __str__(self, *args, **kwargs):
    sResult = str(self.value) + " ->"
    return sResult
  # -----------------------------------------------------------------------------
  def __repr__(self, *args, **kwargs):
    return self.__str__()
  # -----------------------------------------------------------------------------
# =========================================================================================



# =========================================================================================
class LinkedList(object):
  # -----------------------------------------------------------------------------
  def __init__(self):
    super(LinkedList, self).__init__()
    self.first = None
    self.last  = None
  # -----------------------------------------------------------------------------
  def append(self, p_oValue):  
    oNewItem = LinkedListItem(p_oValue)
    self.append_item(oNewItem)
  # -----------------------------------------------------------------------------
  def append_item(self, p_oItem):
    if self.last is None:
      self.first = p_oItem
      self.last  = self.first
    else:    
      self.last.next  = p_oItem
      self.last       = p_oItem
  # -----------------------------------------------------------------------------
  def iterate(self):
    oCurrent = self.first
    while oCurrent is not None:
      yield oCurrent
      oCurrent = oCurrent.next
  # -----------------------------------------------------------------------------      
  def __iter__(self):
    return self.iterate()      
  # -----------------------------------------------------------------------------
# =========================================================================================





# =========================================================================================
class LinkedListDict(LinkedList):
  # -----------------------------------------------------------------------------
  def __init__(self, p_dSourceDict=None):
    super(LinkedListDict, self).__init__()
    self.first = None
    self.last  = None
    
    if p_dSourceDict is not None:
      for sKey, sValue in p_dSourceDict.items():
        self[sKey] = sValue
  # -----------------------------------------------------------------------------
  def __setitem__(self, p_sKey, p_oValue):  
    oFoundItem = None
    for oItem in self.iterate():
      if oItem.value[0] == p_sKey:
        oFoundItem = oItem
        break
      
    if oFoundItem is None:
      oItem = LinkedListItem([p_sKey, p_oValue])
      self.append_item(oItem)
    else:
      oFoundItem.value[1] = p_oValue
  # -----------------------------------------------------------------------------
  def __getitem__(self, p_sKey):
    oValue = None

    for oItem in self.iterate():
      if oItem.value[0] == p_sKey:
        oValue = oItem.value[1]
        break
      
    return oValue
  # -----------------------------------------------------------------------------
  ''' 
  def __next__(self):
      if self.num > self.end:
          raise StopIteration
      else:
          self.num += 1
          return self.num - 1
  '''  
  # -----------------------------------------------------------------------------
  def keys(self):
    return [oItem.value[0] for oItem in self.iterate()]
  # -----------------------------------------------------------------------------
  def __str__(self, *args, **kwargs):
    sResult = ""
    for oItem in self.iterate():
      sResult += f"{oItem.value[0]}:{oItem.value[1]},"
    sResult = "{\n" + sResult[:-1] + "}"
    return sResult
  # -----------------------------------------------------------------------------
  def __repr__(self, *args, **kwargs):
    return self.__str__()
  # -----------------------------------------------------------------------------
# =========================================================================================

  

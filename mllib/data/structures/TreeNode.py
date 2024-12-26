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
class CTreeNodeList(list):
  # -----------------------------------------------------------------------------
  def __init__(self):
    # ................................................................
    # // Fields \\    
    super(CTreeNodeList, self).__init__()
    # ................................................................
  # -----------------------------------------------------------------------------
  def FindByID(self, p_nID):
    oFound = None
    for oNode in self:
      if oNode.ID == p_nID:
        oFound = oNode
        break
        
    return oFound
  # -----------------------------------------------------------------------------
# =========================================================================================







# =========================================================================================
class CTreeNode(object):
  ROOT_SYMBOL      = "/"
  PATH_SEPARATOR   = "/"
  STR_USE_BRACKETS = True
  # ............................................................
  @property
  def Tree(self):
    if self.Parent is None:
      return self
    else:
      return self.Parent.Tree 
  # ............................................................
  @property
  def Root(self):
    if self.Parent is None:
      return self    
    else:
      return self.Parent.Root      
  # ............................................................
  @property
  def IsRoot(self):
    return self.Parent is None
  # ............................................................
  @property
  def Children(self):
    return self.children
  # ............................................................
  @property
  def Level(self):
    if self.Parent is None:
      return 0
    else:
      return self.Parent.Level + 1
  # ............................................................
  @property
  def Path(self):
    if self.Parent is None:
      sResult = CTreeNode.ROOT_SYMBOL
    else:
      sResult = self.Parent.Path 
      if not self.Parent.IsRoot:
        sResult += CTreeNode.PATH_SEPARATOR
      sResult += str(self.ID)
    return sResult
  # ............................................................              



  # -----------------------------------------------------------------------------
  def __init__(self, p_nID):
    # ................................................................
    # // Fields \\    
    self.ID           = p_nID
    self.Parent       = None
    self.children     = CTreeNodeList()
    self.Value        = None
    # ................................................................
  # -----------------------------------------------------------------------------
  def FindChildByID(self, p_nID):
    return self.children.FindByID(p_nID)
  # -----------------------------------------------------------------------------
  def AddChild(self, p_nID):
    oNewNode = CTreeNode(p_nID)
    return self.AddChildNode(oNewNode)
  # -----------------------------------------------------------------------------
  def AddChildNode(self, p_oChildNode):
    p_oChildNode.Parent = self
    self.Tree.Nodes.append(p_oChildNode)
    self.children.append(p_oChildNode)
    #self.childrenIDs.append(p_oChildNode.ID)
    
    if CTree.IS_MAINTAINING_LEAF_LIST:
      self.Tree.Leafs.append(p_oChildNode)
      if self in self.Tree.Leafs:
        self.Tree.Leafs.remove(self)
        
    return p_oChildNode
  # -----------------------------------------------------------------------------
  def RemoveChildNode(self, p_oChildNode):
    self.Tree.Nodes.remove(p_oChildNode)
    self.children.remove(p_oChildNode)
    p_oChildNode.Parent = None    
   
    if CTree.IS_MAINTAINING_LEAF_LIST:
      self.Tree.Leafs.remove(self)
      if len(self.children) == 0:
        self.Tree.Leafs.append(self)
    
    return p_oChildNode
  # -----------------------------------------------------------------------------
  def __str__(self)->str:
    sResult = self.Path
    if CTreeNode.STR_USE_BRACKETS:
      sResult = "[" + sResult + "]"
    return sResult
  # -----------------------------------------------------------------------------
  def __repr__(self)->str:
    sResult = self.__str__()
    if self.Value is not None:
      sResult += ":%s" % str(self.Value) 
          
    return sResult
  # -----------------------------------------------------------------------------
# =========================================================================================














# =========================================================================================
class CTree(CTreeNode):
  IS_MAINTAINING_LEAF_LIST = False
  
  # -----------------------------------------------------------------------------
  def __init__(self):
    super(CTree, self).__init__(0)
    # ................................................................
    # // Fields \
    self.Nodes     = CTreeNodeList()
    self.Nodes.append(self)
    self.Leafs     = CTreeNodeList()
    # ................................................................
  # -----------------------------------------------------------------------------
  @property
  def NodeCount(self):
    return len(self.Nodes)
  # -----------------------------------------------------------------------------
  @property
  def LeafsCount(self):
    return len(self.Leafs)
  # -----------------------------------------------------------------------------
  def TraverseBFS(self):
              #Level, UID 0 == None, UID 1 == Root, self
    oQueue = [[-1, -1, 0, self]]
    nNextUID = 1

    while True:
      if len(oQueue) == 0:
        break
      nLevel, nParentUID, nNodeUID, oNode = oQueue.pop(0)
      if nLevel >= 0:
        yield nLevel, nParentUID, nNodeUID, oNode.Parent, oNode
      
      for oChildNode in oNode.children:
        oQueue.append([nLevel+1, nNodeUID, nNextUID, oChildNode])
        nNextUID += 1
  # -----------------------------------------------------------------------------
  def StructureAndValues(self, p_oProgress=None):
    oTreeStructure = []
    oValueList = []
    for nIndex, (nLevel, nParentUID, nNodeUID, oParentNode, oNode) in enumerate(self.TraverseBFS()):
      if oParentNode is None:
        oTreeStructure.append([nLevel, nParentUID, nNodeUID, 0, oNode.ID, nIndex])
      else:
        oTreeStructure.append([nLevel, nParentUID, nNodeUID, oParentNode.ID, oNode.ID, nIndex])
      oValueList.append(oNode.Value)
      if p_oProgress is not None:
        p_oProgress.update(1)
        
    nTreeStructure = np.array(oTreeStructure, dtype=np.int32)
    return nTreeStructure, oValueList    
  # -----------------------------------------------------------------------------
  def Grow(self, p_nStructure, p_oValues=None, p_oProgress=None):
    oPrevLevelNodes   = None
    oThisLevelNodes   = [[-1, self]]
    nCurrentLevel = -1
    nCurrentPUID  = -1
    
    for nNodeStruct in p_nStructure:
      nLevel, nPUID, nNUID, nParentID, nID, nValueIndex = nNodeStruct
        
      if nLevel != nCurrentLevel:
        nCurrentLevel   = nLevel
        oPrevLevelNodes = oThisLevelNodes
        oThisLevelNodes = []  
        nCurrentPUID = -1
      
      if nPUID != nCurrentPUID:
        nCurrentPUID = nPUID
        for nIndex,(nUID, oParentNode) in enumerate(oPrevLevelNodes):
          if nPUID == nUID:
            oPrevLevelNodes.pop(nIndex)
            break 
        
          
      oNewNode = CTreeNode(nID)
      oParentNode.AddChildNode(oNewNode)
      if p_oValues is not None:
        oNewNode.Value = p_oValues[nValueIndex]
      
      oThisLevelNodes.append([nNUID, oNewNode])
      if p_oProgress is not None:
        p_oProgress.update(1)  
      
  # -----------------------------------------------------------------------------
# =========================================================================================    





# // Unit Testing \\
def UnitTesting(p_oTree, p_oTestNodes):
  oNode6, oNode7, oNode10 = p_oTestNodes
  
  # // Unit Testing \\
  assert str(p_oTree) == "[>]"
  
  assert str(oNode6) == "[>2.6]"
  print("Node6", oNode6)
  
  assert p_oTree.Leafs[0] == oNode7
  assert p_oTree.Leafs[-1] == oNode10
  print("Leaf Nodes:", p_oTree.Leafs)
  
  print("Tree Nodes:", p_oTree.Nodes)
  assert len(p_oTree.Nodes) == 11 
    
    
def UnitTestingData(p_oTree):
  oNode1 = p_oTree.AddChild(10)
  oNode2 = p_oTree.AddChild(2)
  oNode7 = p_oTree.AddChild(7)
  oNode8 = p_oTree.AddChild(80)
  
  oNode1.AddChild(3)
  oNode4 = oNode1.AddChild(4)

  oNode2.AddChild(5)
  oNode6 = oNode2.AddChild(6)
  oNode6.Value = "Value6"
  
  
  
  oNode8.AddChild(9)
  oNode10 = oNode6.AddChild(10)
  oNode10.Value = "Value10"

  
  return oNode6, oNode7, oNode10
  

  
    
if __name__ == "__main__":
  import numpy as np
  CTree.IS_MAINTAINING_LEAF_LIST = True
  CTreeNode.PATH_SEPARATOR = "."
  CTreeNode.ROOT_SYMBOL = ">"
  oTree = CTree()
  p_oTestNodes = UnitTestingData(oTree)
  UnitTesting(oTree, p_oTestNodes)
  
  
  nStructure, oValues = oTree.StructureAndValues()
  print(nStructure)
  
  oNewTree = CTree()
  oNewTree.Grow(nStructure, oValues)
  
  nStructure, oValues = oNewTree.StructureAndValues()
  oValues = np.array(oValues)[..., np.newaxis]
  
  nInfo = np.concatenate([  nStructure[:,:5], oValues[nStructure[:, 5]]
                            ,nStructure[:,5][...,np.newaxis]  
                         ], axis=1)
                  
  
  print(nInfo)
  


  
  

  
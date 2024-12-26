from models.resnet import CResNet, CResidualModule, ResidualModule
from models.densenet import DenseNet, DenseNetBlock, DenseNetTransitionModule
from models.excnn.exresnet import CExResidualModule
from models.excnn import ExResidualModule
from models.excnn import ExDenseNetBlock, ExDenseNetTransitionModule
from models.excnn import RExplaiNetModule
from models.resnet.rexplainet import CRExplaiNetModule
from models.excnn.exresidual_model_clipped import RExplaiNetModuleClipped
from models.excnn.exdense_module_c import ExDenseNetBlockClipped, ExDenseNetTransitionModuleClipped

# ----------------------------------------------------------------------------------------------------------------------
def PredictorModel(dSetup):
  '''
    Model factory method
  '''
  print("(>) Creating model")

  sArchitectureCode = dSetup["code"]
  sDataset   = dSetup["dataset"]
  sBasename  = dSetup["basename"].upper()
  dConfig    = dSetup["config"]

  print(f"Architecture:{sArchitectureCode}")
  if "DENSE" in sBasename:
    if not set_dense_module(sBasename, sDataset):
      print("  |__ Using vanilla densenet module")
    oCNNModel = DenseNet(dConfig)
  else:
    if not set_residual_module(sBasename, sDataset):
      print("  |__ Using vanilla residual module")
    print(f"(*) Layers")
    oCNNModel = CResNet(dConfig)

  return oCNNModel
# ----------------------------------------------------------------------------------------------------------------------
def set_dense_module(experiment_basename, dataset_name):
  bIsExplainable = False
  sInhibitorType = experiment_basename[0]
  DenseNet.DenseModuleClass = DenseNetBlock
  DenseNet.DenseTransitionClass = DenseNetTransitionModule
  if not experiment_basename.upper().startswith("DENSENET"):
    if (sInhibitorType == "X") or (sInhibitorType == "L"):
      DenseNet.DenseModuleClass = ExDenseNetBlock
      DenseNet.DenseTransitionClass = ExDenseNetTransitionModule
      bIsExplainable = True
    elif sInhibitorType == "C":
      DenseNet.DenseModuleClass = ExDenseNetBlockClipped
      DenseNet.DenseTransitionClass = ExDenseNetTransitionModuleClipped
      bIsExplainable = True


  return bIsExplainable
# ----------------------------------------------------------------------------------------------------------------------
def set_residual_module(experiment_basename, dataset_name):
  experiment_basename = experiment_basename.upper()

  bIsExplainableResidualModule = True
  if experiment_basename.startswith("RESNET"):
    # Vanilla residual module uses batch normalization on skip connections
    bIsExplainableResidualModule = False
    CResNet.ResidualModuleClass = CResidualModule
    if dataset_name.lower() == "cifar10":
      CResNet.ResidualModuleClass = ResidualModule
  else:
    sInhibitorType = experiment_basename[0]
    if sInhibitorType == "R":
      CResNet.ResidualModuleClass = CExResidualModule
    elif sInhibitorType == "L":
      CResNet.ResidualModuleClass = CRExplaiNetModule
    elif sInhibitorType == "S":
      # Uses batch normalization on skip connections
      CResNet.ResidualModuleClass = RExplaiNetModule
    elif sInhibitorType == "H":
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "E": # Not supported
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "Q": # Not supported
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "N": # Not supported
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "D": # Not supported
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "X":
      # Uses batch normalization on skip connections
      CResNet.ResidualModuleClass = ExResidualModule
    elif sInhibitorType == "G":
      raise f"Inhititor type {sInhibitorType} not support in this release"
    elif sInhibitorType == "C":
      CResNet.ResidualModuleClass = RExplaiNetModuleClipped
    elif sInhibitorType == "T":
      raise f"Inhititor type {sInhibitorType} not support in this release"

  return bIsExplainableResidualModule
# ----------------------------------------------------------------------------------------------------------------------
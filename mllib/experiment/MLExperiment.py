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

import os
import numpy as np
from datetime import datetime
from tensorflow import keras
from mllib.system import CMLSystem, FileStore
from mllib.evaluation import CEvaluateClassification
from mllib.visualization import CPlotTrainingLogs

from .mlexperiment_env_legacy import CMLExperimentEnv, model_code
from .mlexperiment_environment import  MLExperimentEnv


# --------------------------------------------------------------------------------------
# Define a custom sorting function
def _sort_by_last_path_element(folder):
  # Split the path into its components
  components = folder.split(os.pathsep)
  # Extract the last path element
  last_element = components[-1]
  # Extract the numeric part of the last element
  numeric_part = ''.join(filter(str.isdigit, last_element))
  # Convert the numeric part to an integer
  try:
    return int(numeric_part)
  except ValueError:
    # If the numeric part is not convertible to an integer, return a large number
    return float('inf')
# --------------------------------------------------------------------------------------



# =========================================================================================================================
class CMLExperiment:
  # --------------------------------------------------------------------------------------
  def __init__(self, p_oEnvConfig, p_oModel=None, p_sMetrics=[]
                , p_oCostFunction=None, p_oLearningAlgorithm=None, p_bIsRetraining=False):
    # ................................................................
    self.Config           = None
    #if isinstance(p_oEnvConfig, CMLExperimentConfig):
      #self.ModelName        = p_oEnvConfig.ExperimentCode
    if isinstance(p_oEnvConfig, MLExperimentEnv):
      self.Config = p_oEnvConfig.Config
    elif isinstance(p_oEnvConfig, CMLExperimentEnv):
      #self.ModelName        = p_oEnvConfig.ExperimentCode
      if hasattr(p_oEnvConfig, "Config"):
        self.Config           = p_oEnvConfig.Config
      elif isinstance(p_oEnvConfig, dict):
        self.Config = p_oEnvConfig
    else:
      if hasattr(p_oEnvConfig, "Config"):
        self.Config = p_oEnvConfig.Config
      elif isinstance(p_oEnvConfig, dict):
        self.Config = p_oEnvConfig

    assert self.Config is not None, "Incompatible ML Experiment configuration object"

    self.is_showing_step_progress = False

    #else:
    #  self.ModelName        = self.Config["ModelName"]
    self.ModelName = model_code(self.Config)

    if isinstance(p_oEnvConfig, MLExperimentEnv):
      self.ModelFS = p_oEnvConfig.ExperimentFS
    else:
      self.ModelFS          = CMLSystem.Instance().ModelFS.SubFS(self.ModelName)

    self.ParamsFS         = self.ModelFS.SubFS("weights")
    self.LogFS            = self.ModelFS.SubFS("logs")
    self.CheckpointFS     = self.ModelFS.SubFS("checkpoints")
    self.Model            = p_oModel
    self.Metrics          = p_sMetrics
    self.CostFunction     = p_oCostFunction
    if p_oLearningAlgorithm is not None:
      self.Optimizer = p_oLearningAlgorithm.Optimizer
      self.Callbacks = p_oLearningAlgorithm.Callbacks
    else:
      self.Optimizer = None
      self.Callbacks = None
    self._dataset        = None
    self._ts_feed         = None
    self._vs_feed         = None
    self._is_graph_built   = False
    self._has_loaded_state = False
    self.IsRetraining   = p_bIsRetraining
    self.ProcessLog       = None
    self.LearningHistory  = None

    self.generator = None
    
    self._modelStateFolder = self.ModelFS.SubPath("state")

    self._start_train_time = None
    self._end_train_time = None

    self._currentModelFolder = None
    self._currentModelStateFolder = None
    self._currentModelLogFolder = None
    # ................................................................

  # --------------------------------------------------------------------------------------
  @property
  def model_state_folder(self):
    return self._modelStateFolder
  # --------------------------------------------------------------------------------------
  @property
  def IsPretrained(self):
    return self.ModelFS.SubFS("state").Exists("saved_model.pb")
  # --------------------------------------------------------------------------------------
  @classmethod
  def Restore(cls, p_sModelName, p_oModel=None):
    oConfig = {"ModelName": p_sModelName}
    oExperiment =  CMLExperiment(oConfig, p_oModel)
    return oExperiment.Load()
  # --------------------------------------------------------------------------------------
  @property
  def Dataset(self):
    return self._dataset
  # --------------------------------------------------------------------------------------
  @Dataset.setter
  def Dataset(self, p_oDataset):
    self._dataset = p_oDataset
    if self._ts_feed is None:
      self._ts_feed = (self._dataset.TSSamples, self._dataset.TSLabels)
    if self._vs_feed is None:
      self._vs_feed = (self._dataset.VSSamples, self._dataset.VSLabels)
  # --------------------------------------------------------------------------------------
  @property
  def TrainingSet(self):
    return self._ts_feed
  # --------------------------------------------------------------------------------------
  @TrainingSet.setter
  def TrainingSet(self, p_oTS):  
    self._ts_feed = p_oTS
  # --------------------------------------------------------------------------------------
  @property    
  def ValidationSet(self):
    return self._vs_feed
  # --------------------------------------------------------------------------------------
  @ValidationSet.setter    
  def ValidationSet(self, p_oVS):
    self._vs_feed = p_oVS
  # --------------------------------------------------------------------------------------
  def Summary(self):
    if not self._is_graph_built:
      for oSamples, oTarget in self._ts_feed.take(1):
        # We recall one batch/sample to create the graph and initialize the model parameters
        y = self.Model(oSamples)
        break
              
      self._is_graph_built = True
      
    self.Model.summary()
  # --------------------------------------------------------------------------------------
  def _timing_info(self):
    # Timings
    nEpochs = self.Config["Training.MaxEpoch"]
    dDiff = self._end_train_time - self._start_train_time
    nElapsedHours = dDiff.total_seconds() / 3600
    nSecondsPerEpoch = dDiff.total_seconds() / nEpochs
    dTiming = { "StartTime": self._start_train_time, "EndTime": self._end_train_time,
                "ElapsedHours": nElapsedHours, "SecondsPerEpoch": nSecondsPerEpoch  }
    return dTiming
  # --------------------------------------------------------------------------------------
  def Save(self):
    self.Model.save(self._modelStateFolder)
    if self.LearningHistory is not None:
      self.LogFS.Serialize("keras_learning_history.pkl", self.LearningHistory)
      dTiming = self._timing_info()
      self.LogFS.obj.save(dTiming, "timing_info.pkl")
      dTiming["StartTime"] = dTiming["StartTime"].strftime('%Y-%m-%dT%H:%M:%S')
      dTiming["EndTime"]  = dTiming["EndTime"].strftime('%Y-%m-%dT%H:%M:%S')
      self.ModelFS.json.save(dTiming, f"timing_info_{self._end_train_time.strftime('%Y-%m-%dT%H%M%S')}.json", is_sorted_keys=False)
      
    # //TODO: Keep cost function names and other learning parameters for evaluation
  # --------------------------------------------------------------------------------------
  def Load(self, p_bLastCheckpoint=False, model_root_folder=None):
    self._currentModelFolder = self.ModelFS.base_folder
    self._currentModelStateFolder = self._modelStateFolder
    self._currentModelLogFolder = FileStore(self.LogFS.base_folder)
    if model_root_folder is not None:
      self._currentModelFolder = model_root_folder
      self._currentModelStateFolder = os.path.join(model_root_folder, "state")
      self._currentModelLogFolder = FileStore(model_root_folder).subfs("logs")

    print("Loading saved state from ", self._currentModelStateFolder)
    self.Model = keras.models.load_model(self._currentModelStateFolder)
    self.LearningHistory = (self._currentModelLogFolder.obj.load("keras_learning_history.pkl"))
    self._has_loaded_state = True
    
    if p_bLastCheckpoint:
      self.LoadModelParams(p_bLastCheckpoint=True)
    
    return self.Model
  # --------------------------------------------------------------------------------------
  def PlotLearningHistory(self):
    oTrainingLogPlot = CPlotTrainingLogs(self.LearningHistory)
    # //TODO: Support for different metric
    oTrainingLogPlot.Show(self.ModelName, p_sMetricName="average_accuracy"
                            , p_oCostFunction=self.CostFunction)          
  # --------------------------------------------------------------------------------------
  def SaveModelParams(self):
    if self.Model is not None:
      self.Model.save_weights(self.ParamsFS.SubPath("model_params"))
  # --------------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------------
  @property
  def CheckPointPaths(self):
    oCheckpointFS = self.CheckpointFS
    if self._currentModelFolder is not None:
      if self._currentModelFolder != self.ModelFS.base_folder:
        oCheckpointFS = FileStore(self._currentModelFolder).subfs("checkpoints")

    sCheckpointFolders =  oCheckpointFS.Files("*.index", True, p_tSortFilenamesAs=int)
    sCheckpointFolders = sorted(sCheckpointFolders, key=_sort_by_last_path_element)
    return sCheckpointFolders
  # --------------------------------------------------------------------------------------
  def unload_model(self):
    del self.Model
    self.Model = None
  # --------------------------------------------------------------------------------------
  def LoadModelParams(self, p_sCheckPointPath=None, p_bLastCheckpoint=False, is_ignoring_not_found=False):
    sTargetPath = None
    sCheckPointPaths = self.CheckPointPaths
    if p_sCheckPointPath is not None:
      if p_sCheckPointPath in sCheckPointPaths:
        sTargetPath = p_sCheckPointPath
      else:
        raise Exception("Model params not found in checkpoint path %s" % p_sCheckPointPath)
    elif p_bLastCheckpoint:
      if len(sCheckPointPaths) > 0:
        sTargetPath = sCheckPointPaths[-1]
      else:
        raise Exception("No checkpoints are saved in %s" % self.CheckpointFS.BaseFolder)
    else:      
      if self.ParamsFS.IsEmpty:
        raise Exception("Model params not found in %s" % self.ParamsFS.BaseFolder)
      else:
        sTargetPath = self.ParamsFS.SubPath("model_params")

    if sTargetPath is not None:
      if is_ignoring_not_found:
        self.Model.load_weights(sTargetPath)
        '''
        nLogLevel = tf.get_logger().getEffectiveLevel()
        try:
          tf.get_logger().setLevel(logging.CRITICAL) #This does not suppress warning during loading of models
          
        finally:
          tf.get_logger().setLevel(nLogLevel)
        '''
      else:
        self.Model.load_weights(sTargetPath)
      print("Loaded weights from %s" % sTargetPath)
      self._has_loaded_state = True
  # --------------------------------------------------------------------------------------
  def TransferModelParamsTo(self, p_oNewModel, p_nInputShape, p_oMetrics=[]):
    self.SaveModelParams()
    del self.Model
    self.Model = p_oNewModel
    self.Model.build(input_shape=p_nInputShape)
    self.LoadModelParams()
    self.Model.compile(metrics=p_oMetrics)#, run_eagerly = True)
    return self.Model
  # --------------------------------------------------------------------------------------
  def Train(self):
    if (not self.IsPretrained) or self.IsRetraining:
      self._start_train_time = datetime.now()
      self.Model.compile(loss=self.CostFunction, optimizer=self.Optimizer, metrics=self.Metrics)
      if CMLSystem.Instance().Params["IsDebuggable"]:
        self.Model.run_eagerly = True

      if self.is_showing_step_progress:
        nVerbose = 1
      else:
        nVerbose = 2

      nEpochs = self.Config["Training.MaxEpoch"]

      if self.generator is not None:
        self.ProcessLog = self.Model.fit_generator(generator=self.generator,
                                                   epochs=nEpochs,
                                                   callbacks=self.Callbacks,
                                                   validation_data=self._vs_feed,
                                                   verbose = nVerbose )
      else:
        if "Training.StepsPerEpoch" in self.Config:
          self.ProcessLog = self.Model.fit( self._ts_feed
                                              ,batch_size=self.Config["Training.BatchSize"]
                                              ,epochs=nEpochs
                                              ,validation_data=self._vs_feed
                                              ,callbacks=self.Callbacks
                                              ,steps_per_epoch=self.Config["Training.StepsPerEpoch"]
                                              ,verbose=nVerbose )
        else:
          self.ProcessLog = self.Model.fit( self._ts_feed
                                              ,batch_size=self.Config["Training.BatchSize"]
                                              ,epochs=nEpochs
                                              ,validation_data=self._vs_feed
                                              ,callbacks=self.Callbacks
                                              ,verbose=nVerbose )
      self._end_train_time = datetime.now()
      self.LearningHistory = self.ProcessLog.history
      self.Save()
      self._is_graph_built = True
    else:
      self.Load()
      
    return self.Model
  # --------------------------------------------------------------------------------------
  def EvaluateClassifier(self, p_nTrueClassLabels=None, p_oSampleFeed=None, p_bIsEvaluatingUsingKeras=False, is_printing=True):
    if p_oSampleFeed is None:
      p_oSampleFeed = self._vs_feed
    nLoss, nAccuracy = (None, None)
    if p_bIsEvaluatingUsingKeras:
      oTestResults = self.Model.evaluate(p_oSampleFeed, verbose=1)
      nLoss, nAccuracy = oTestResults
      print("Evaluation: Loss:%.6f - Accuracy:%.6f"  % (nLoss, nAccuracy))
    
    # ... // Evaluate \\ ...
    nVerbose=0
    if is_printing:
      nVerbose=1

    nPrediction = self.Model.predict(p_oSampleFeed, verbose=nVerbose)
    if isinstance(nPrediction, tuple):
      nPredictedClassProbabilities, nPredictedClassLabels, nIDs = nPrediction
    else:
      nPredictedClassProbabilities = nPrediction
      nPredictedClassLabels = np.argmax(nPredictedClassProbabilities, axis=1)
      nIDs = None
    
    if p_nTrueClassLabels is None:
      p_nTrueClassLabels = self._dataset.VSLabels

    if is_printing:
      return self.EvaluationReport(p_nTrueClassLabels, nPredictedClassLabels, nIDs), nLoss, nAccuracy
    else:
      return self.evaluation(p_nTrueClassLabels, nPredictedClassLabels), nLoss, nAccuracy
  # --------------------------------------------------------------------------------------
  def evaluation(self, true_class_labels, predicted_class_labels):
    p_nTrueClassLabels = true_class_labels.reshape(-1)
    p_nPredictedClassLabel = predicted_class_labels.reshape(-1)
    oEvaluator = CEvaluateClassification(p_nTrueClassLabels, p_nPredictedClassLabel)

    return oEvaluator
  # --------------------------------------------------------------------------------------
  def EvaluationReport(self, p_nTrueClassLabels, p_nPredictedClassLabel, p_nID=None):
    p_nTrueClassLabels = p_nTrueClassLabels.reshape(-1)
    p_nPredictedClassLabel = p_nPredictedClassLabel.reshape(-1)
    
    oEvaluator = CEvaluateClassification(p_nTrueClassLabels, p_nPredictedClassLabel)
    oEvaluator.PrintConfusionMatrix()
    print(f"Accuracy                      : {oEvaluator.Accuracy:.4f}")
    print(f"Per Class Recall (Accuracy)   : {oEvaluator.Recall}")
    print(f"Per Class Precision           : {oEvaluator.Precision}") 
    print(f"AverageF1Score                : %.4f" % oEvaluator.AverageF1Score)
        
    
    if p_nID is not None:
      bMissclassifiedFlags = (p_nTrueClassLabels!= p_nPredictedClassLabel)
      print(f"Missclassified Samples: {np.sum(bMissclassifiedFlags)}/{p_nTrueClassLabels.shape[0]}")
      
      nMissTrue      = p_nTrueClassLabels[bMissclassifiedFlags]
      nMissPredicted = p_nPredictedClassLabel[bMissclassifiedFlags]
      nMissIDs       = p_nID[bMissclassifiedFlags]
      for i, nID in enumerate(nMissIDs):
        print(f"  |__ Sample#{int(nID):07d} True:{int(nMissTrue[i])} != {int(nMissPredicted[i])}")

    return oEvaluator
  # --------------------------------------------------------------------------------------
# =========================================================================================================================



  
  
    
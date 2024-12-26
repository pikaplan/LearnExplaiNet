import argparse
from keras import metrics, losses

import mllib as ml

from mllib.experiment import CMLExperiment, MLExperimentEnv
from mllib.learning import CLearningAlgorithm, CLearningBestStateSaver

from data import select_dataset
from models import PredictorModel

# -------------------------- Usage example --------------------------
# > python XTrainMulti.py -g MNIST-GROUP -m LREXPLAINET22_MNIST -n 64 -f 3 -ds mnist
# -------------------------------------------------------------------


oParser = argparse.ArgumentParser(description="Description of your script")
oParser.add_argument("-g", "--model_group", type=str, default="EXRESNET_FMNIST", help="The prefix of the models name in an experiments group.")
oParser.add_argument("-m", "--model", type=str, default="EREXPLAINET18_FMNIST", help="The prefix of the models name in an experiments group.")
oParser.add_argument("-n", "--number_var", type=str, default="64", help="The number of experiment variation in the form {major.minor}.")
oParser.add_argument("-ds", "--dataset", type=str, default="kmnist", help="Dataset to use for the experiments.")
oParser.add_argument("-rnd", "--random_seed", type=int, default=2000, help="Random seed for reproducibility.")
oParser.add_argument("-f", "--fold_number", type=int, default=None, help="Fold number of the experiment, for k-Fold or k random seed.")
oParser.add_argument("-fd", "--is_debug", action='store_true', help="Enable debug mode.")
oParser.add_argument("-ft", "--is_retraining", action='store_true', help="Force retraining of model, overwriting existing saved states.")
oParser.add_argument("-famp", "--is_auto_mixed_precision", action='store_true', default=False, help="Enable auto mixed precision in tensorflow")
oParser.add_argument("-fdet", "--is_deterministic", action='store_true', default=True, help="Force deterministic behaviour of tensorflow")
oParser.add_argument("-ftf32", "--is_using_tensorfloat32", action='store_true', default=False, help="Enable tensorfloat32 acceleration.")
oParser.add_argument("-fft", "--is_fft_cudnn", action='store_true', default=True, help="Select FFT backward filter algorithm for cuDNN.")
switches = oParser.parse_args()

CONFIG, oPrintQueue, dExperimentSpec = MLExperimentEnv.preload_config("ZConfig", switches.model_group, switches.model, switches.number_var, switches.fold_number)
if "Experiment.RandomSeed" in CONFIG:
  switches.random_seed = CONFIG["Experiment.RandomSeed"]

oPrintQueue.extend(ml.CMLSystem.prepare_environment(is_fft_convolution_algorithm=(switches.is_fft_cudnn) or (switches.dataset.lower() == "mnist"),
                                                    is_auto_mixed_precision=switches.is_auto_mixed_precision))

oFileSys = ml.FileSystem(r"ZConfig", r"ZModels", r"ZData", model_group=switches.model_group)

ml.CMLSystem.RANDOM_SEED = switches.random_seed
ml.CMLSystem.IS_DETERMINISTIC = switches.is_deterministic
ml.CMLSystem.IS_USING_TENSORFLOAT32 = switches.is_using_tensorfloat32
oSys = ml.CMLSystem.Instance(switches).with_file_system(oFileSys)

# _____ | Hyperparameters | ______
oEnv = MLExperimentEnv(oFileSys, switches.model, variation=switches.number_var, fold_number=switches.fold_number, experiment_config=CONFIG)
oEnv.move_log(oSys.start_timestamp, oSys.log_start_filename_to_move)
for s in oPrintQueue:
  print(s)

oSys = ml.CMLSystem

#CONFIG = oEnv.Config

## _____ | Data | ______
oDataset, oData = select_dataset(switches.dataset, oFileSys, CONFIG)

# _____ | Model / Training | ______
dSetup = { "code": switches.model, "basename": oEnv.BaseName.upper(), "dataset": switches.dataset, "config": CONFIG }
oCNNModel = PredictorModel(dSetup)

oMetrics = [metrics.CategoricalAccuracy(name="average_accuracy", dtype=None)]
oCostFunction = losses.CategoricalCrossentropy()
oLearningAlgorithm = CLearningAlgorithm(CONFIG)
oStateSaver = CLearningBestStateSaver(oEnv.ExperimentFS, p_sMetric="average_accuracy")

# ... // Train \\ ...
oExperiment = CMLExperiment(oEnv, oCNNModel, oMetrics, oCostFunction, oLearningAlgorithm, p_bIsRetraining=switches.is_retraining)
oExperiment.TrainingSet = oData.TSFeed
oExperiment.ValidationSet = oData.VSFeed
oExperiment.Callbacks.append(oStateSaver.Callback)

if (not oExperiment.IsPretrained) or oExperiment.IsRetraining:
  oCNNModel = oExperiment.Train()
else:
  oCNNModel = oExperiment.Load(True)

# ... // Evaluate \\ ...
oExperiment.Dataset = oDataset
oExperiment.EvaluateClassifier()
oExperiment.Model.summary()
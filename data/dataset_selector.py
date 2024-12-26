from data.mnist import CMNISTDataSet, CMNISTDataFeed
from data.fashion_mnist import FashionMNISTDataset, FashionMNISTDataFeed
from data.kmnist import KMNISTDataset, KMNISTDataFeed
from data.oracle_mnist import OracleMNISTDataset, OracleMNISTDataFeed
from data.cifar10 import CIFAR10Dataset, CCIFAR10DataFeed


def select_dataset(dataset_name, filestore, config=None):
  if config is None:
    config = {"InputShape": [28,28,1], "ClassCount":10, "Training.BatchSize": 128,
                "Validation.BatchSize": 128, "Prediction.BatchSize": 200, "DatasetName": dataset_name}
  ## _____ | Data | ______
  oDataset, oDataFeeds= None, None
  if dataset_name == "mnist":
    oDataset = CMNISTDataSet(filestore.datasets)
    assert oDataset.Name.upper() == config["DatasetName"].upper(), "Configuration file is not for MNIST."
    oDataset.Load()
    oDataset.PrintInfo()
    oDataFeeds = CMNISTDataFeed(oDataset, config)
  elif (dataset_name == "fmnist") or (dataset_name == "fashion_mnist"):
    oDataset = FashionMNISTDataset(filestore.datasets.base_folder)
    assert oDataset.Name.upper() == config["DatasetName"].upper(), "Configuration file is not for Fashion MNIST."
    oDataset.load()
    oDataset.PrintInfo()
    oDataFeeds = FashionMNISTDataFeed(oDataset, config)
  elif (dataset_name == "kmnist") or (dataset_name == "kuzushiji_mnist"):
    oDataset = KMNISTDataset(filestore.datasets.base_folder)
    assert oDataset.Name.upper() == config["DatasetName"].upper(), "Configuration file is not for Kuzushiji MNIST."
    oDataset.load()
    oDataset.PrintInfo()
    oDataFeeds = KMNISTDataFeed(oDataset, config)
  elif (dataset_name == "omnist") or (dataset_name == "oracle_mnist"):
    oDataset = OracleMNISTDataset(filestore.datasets.base_folder)
    assert oDataset.Name.upper() == config["DatasetName"].upper(), "Configuration file is not for Oracle MNIST."
    oDataset.load()
    oDataset.PrintInfo()
    oDataFeeds = OracleMNISTDataFeed(oDataset, config)
  elif dataset_name == "cifar10":
    config["InputShape"] = [32,32,3]
    oDataset = CIFAR10Dataset(filestore.datasets.base_folder)
    assert oDataset.Name.upper() == config["DatasetName"].upper(), "Configuration file is not for CIFAR10."
    oDataset.load()
    oDataset.PrintInfo()
    oDataFeeds = CCIFAR10DataFeed(oDataset, config)

  return oDataset, oDataFeeds
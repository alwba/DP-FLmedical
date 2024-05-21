from torchsummary import summary

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.loaders import DatasetLoaderMNIST, DatasetLoaderCOVIDx, DatasetLoaderDiabetes, \
    DatasetLoaderHeartDisease
from classifiers import MNIST, CovidNet, CNN, Diabetes, HeartDisease
from logger import logPrint
from client import Client
import aggregators as agg

import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import random
import torch
import time

import os
import shutil

def __experimentOnMNIST(config):
    dataLoader = DatasetLoaderMNIST().getDatasets
    classifier = MNIST.Classifier
    __experimentSetup(config, dataLoader, classifier)


def __experimentOnCONVIDx(config, model='COVIDNet'):
    datasetLoader = DatasetLoaderCOVIDx().getDatasets
    if model == 'COVIDNet':
        classifier = CovidNet.Classifier
    elif model == 'resnet18':
        classifier = CNN.Classifier
    else:
        raise Exception("Invalid Covid model name.")
    __experimentSetup(config, datasetLoader, classifier)


def __experimentOnDiabetes(config):
    datasetLoader = DatasetLoaderDiabetes(config.requireDatasetAnonymization).getDatasets
    classifier = Diabetes.Classifier
    __experimentSetup(config, datasetLoader, classifier)


def __experimentOnHeartDisease(config):
    dataLoader = DatasetLoaderHeartDisease(config.requireDatasetAnonymization).getDatasets
    classifier = HeartDisease.Classifier
    __experimentSetup(config, dataLoader, classifier)


def __experimentSetup(config, datasetLoader, classifier):
    errorsDict = dict()
    for aggregator in config.aggregators:
        if config.privacyPreserve is not None:
            name = aggregator.__name__.replace("Aggregator", (" with DP" if config.privacyPreserve else ""))
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier,
                                               aggregator, config.privacyPreserve)
        else:
            name = aggregator.__name__.replace("Aggregator", "")
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier, aggregator,
                                               useDifferentialPrivacy=False)
            logPrint("TRAINING {} with DP".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier, aggregator,
                                               useDifferentialPrivacy=True)

    if config.plotResults:
        plt.figure()
        i = 0
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan',
                  'tab:purple', 'tab:pink', 'tab:olive', 'tab:brown', 'tab:gray']
        for name, err in errorsDict.items():
            plt.plot(err.numpy(), color=colors[i])
            i += 1
        plt.legend(errorsDict.keys())
        plt.show()


def __runExperiment(config, datasetLoader, classifier, aggregator, useDifferentialPrivacy):
    trainDatasets, testDataset = datasetLoader(config.percUsers, config.labels, config.datasetSize)
    clients = __initClients(config, trainDatasets, useDifferentialPrivacy)
    # Requires model input size update due to dataset generalisation and categorisation
    if config.requireDatasetAnonymization:
        classifier.inputSize = testDataset.getInputSize()
    model = classifier().to(config.device)
    aggregator = aggregator(clients, model, config.rounds, config.device, config.exp_name)

    return aggregator.trainAndTest(testDataset)


def __initClients(config, trainDatasets, useDifferentialPrivacy):
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    logPrint("Creating clients...")
    clients = []
    for i in range(usersNo):
        clients.append(Client(idx=i + 1,
                              trainDataset=trainDatasets[i],
                              epochs=config.epochs,
                              batchSize=config.batchSize,
                              learningRate=config.learningRate,
                              p=p0,
                              alpha=config.alpha,
                              beta=config.beta,
                              Loss=config.Loss,
                              Optimizer=config.Optimizer,
                              device=config.device,
                              useDifferentialPrivacy=useDifferentialPrivacy,
                              epsilon1=config.epsilon1,
                              epsilon3=config.epsilon3,
                              needClip=config.needClip,
                              clipValue=config.clipValue,
                              needNormalization=config.needNormalization,
                              releaseProportion=config.releaseProportion))

    nTrain = sum([client.n for client in clients])
    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.n / nTrain

    # Create malicious (byzantine) and faulty users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User", client.id, "is faulty.")
        if client.id in config.malicious:
            client.flip = True
            logPrint("User", client.id, "is malicious.")
            client.trainDataset.zeroLabels()
    return clients


def __setRandomSeeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


#   EXPERIMENTS
def experiment(exp):
    def decorator():
        __setRandomSeeds(2)
        logPrint("Experiment {} began.".format(exp.__name__))
        begin = time.time()
        exp()
        end = time.time()
        logPrint("Experiment {} took {}".format(exp.__name__, end - begin))

    return decorator

@experiment
def baseline_onHeartDisease():
    # baseline experiments without DP on 1,3,5,10 clients
    learningRate = 0.0001
    batchSize = 5
    epochs = 10
    rounds = 100

    noOfClients = {'1': torch.tensor([1.]), '3': torch.tensor([1/3, 1/3, 1/3]), '5': torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5]), '10': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])}

    for n, percUsers in noOfClients.items():
        logPrint('Baseline experiment on heart disease with', n, 'clients:')
        noDPconfig = DefaultExperimentConfiguration()#
        noDPconfig.exp_name = 'heartDisease/heartDisease_baseline_' + n + '_clients'
        noDPconfig.Optimizer = torch.optim.Adam
        noDPconfig.aggregators = agg.FA()
        noDPconfig.learningRate = learningRate
        noDPconfig.batchSize = batchSize
        noDPconfig.epochs = epochs
        noDPconfig.rounds = rounds
        noDPconfig.percUsers = percUsers
        noDPconfig.plotResults = False

        path = 'out/' + noDPconfig.exp_name
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        __experimentOnHeartDisease(noDPconfig)

@experiment
def baseline_onDiabetes():
    # baseline experiments without DP on 1,3,5,10 clients
    learningRate = 0.00001
    batchSize = 10
    epochs = 5
    rounds = 50

    noOfClients = {'1': torch.tensor([1.]), '3': torch.tensor([1/3, 1/3, 1/3]), '5': torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5]), '10': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])}

    for n, percUsers in noOfClients.items():
        logPrint('Baseline experiment on diabetes with ', n, 'clients:')
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.exp_name = 'diabetes/diabetes_baseline_' + n + '_clients'
        noDPconfig.Optimizer = torch.optim.Adam
        noDPconfig.aggregators = agg.FA()
        noDPconfig.learningRate = learningRate
        noDPconfig.batchSize = batchSize
        noDPconfig.epochs = epochs
        noDPconfig.rounds = rounds
        noDPconfig.percUsers = percUsers
        noDPconfig.plotResults = False

        path = 'out/' + noDPconfig.exp_name
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        __experimentOnDiabetes(noDPconfig)

@experiment
def customExperiment():
    config = DefaultExperimentConfiguration()
    config.percUsers = torch.tensor([1.])

    config.learningRate = 0.0001
    config.batchSize = 20
    config.epochs = 10
    config.rounds = 100

    # config.requireDatasetAnonymization = True
    __experimentOnDiabetes(config)


customExperiment()

import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from numpy import clip, percentile, array, concatenate, empty

from scipy.stats import laplace

from logger import logPrint


class Client:
    """ An internal representation of a client """

    def __init__(self, epochs, batchSize, learningRate, trainDataset, p, idx, useDifferentialPrivacy,
                 releaseProportion, epsilon, delta, needClip, clipValue, device, Optimizer, Loss,
                 needNormalization, model=None):

        self.name = "client" + str(idx)
        self.device = device

        self.model = model
        self.trainDataset = trainDataset
        self.dataLoader = DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True)
        self.n = len(trainDataset)  # Number of training points provided
        self.p = p  # Contribution to the overall model
        self.id = idx  # ID for the user

        # Used for computing dW, i.e. the change in model before
        # and after client local training, when DP is used
        self.untrainedModel = copy.deepcopy(model).to('cpu') if model else False

        self.opt = None
        self.sim = None
        self.loss = None
        self.Loss = Loss
        self.Optimizer = Optimizer
        self.pEpoch = None
        self.badUpdate = False
        self.epochs = epochs
        self.batchSize = batchSize

        self.learningRate = learningRate
        self.momentum = 0.9
        self.blocked = False

        # DP parameters
        self.useDifferentialPrivacy = useDifferentialPrivacy
        self.epsilon = epsilon
        self.delta = delta
        self.needClip = needClip
        self.clipValue = clipValue
        self.needNormalization = needNormalization
        self.releaseProportion = releaseProportion

    def updateModel(self, model):
        self.model = model.to('cpu')
        if self.Optimizer == optim.SGD:
            self.opt = self.Optimizer(self.model.parameters(), lr=self.learningRate, momentum=self.momentum)
        else:
            self.opt = self.Optimizer(self.model.parameters(), lr=self.learningRate)
        self.loss = self.Loss()
        self.untrainedModel = copy.deepcopy(model).to('cpu')
        torch.cuda.empty_cache()

    # Function to train the model for a specific user
    def trainModel(self):
        self.model = self.model.to(self.device)
        for i in range(self.epochs):
            for iBatch, (x, y) in enumerate(self.dataLoader):
                x = x.to(self.device)
                y = y.to(self.device)
                err, pred = self._trainClassifier(x, y)
            # logPrint("Client:{}; Epoch{}; Batch:{}; \tError:{}"
            #          "".format(self.id, i + 1, iBatch + 1, err))
        torch.cuda.empty_cache()
        self.model = self.model.to('cpu')
        return err, pred

    # Function to train the classifier
    def _trainClassifier(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = self.model(x).to(self.device)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
        return err, pred

    # Function used by aggregators to retrieve the model from the client
    def retrieveModel(self):
        if self.useDifferentialPrivacy:
            return self.__privacyPreserve()
        
        return self.model

    def __addGaussianNoise(self, model_update, sensitivity, epsilon, delta):
        # Calculate the standard deviation of the Gaussian noise
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Generate Gaussian noise
        noise = np.random.normal(loc=0, scale=sigma, size=model_update.shape)
        
        # Add noise to the model update
        noisy_update = model_update + noise
        return noisy_update
    
    def __clipModelUpdate(self, model_update, clip_norm):
        # Compute the L2 norm of the model update
        norm = np.linalg.norm(model_update)
        
        # Clip the model update
        if norm > clip_norm:
            model_update = model_update * (clip_norm / norm)
        
        return model_update
    
    def __privacyPreserve2(self):
        sensitivity = 1.0  # Sensitivity of the function, assuming each update is normalized

        paramArr = nn.utils.parameters_to_vector(self.model.parameters())
        untrainedParamArr = nn.utils.parameters_to_vector(self.untrainedModel.parameters())

        paramChanges = (paramArr - untrainedParamArr).detach().to(self.device)
        logPrint('paramChanges:', paramChanges[:5])
        clippedParamChanges = self.__clipModelUpdate(paramChanges, 5.0)
        logPrint('clippedParamChanges:', clippedParamChanges[:5])
        noisyUpdate = self.__addGaussianNoise(clippedParamChanges, sensitivity, self.epsilon, self.delta)
        logPrint('noisyUpdate:', noisyUpdate[:5])

        nn.utils.vector_to_parameters(noisyUpdate, self.model.parameters())
        
        return self.model.to(self.device)

    # Procedure for implementing differential privacy
    def __privacyPreserve(self):
        logPrint("Privacy preserving for client{} in process..".format(self.id))
        logPrint("epsilon={}".format(self.epsilon))
        gamma = self.clipValue  # gradient clipping value
        s = 2 * gamma  # sensitivity
        Q = self.releaseProportion  # proportion to release

        # The gradients of the model parameters
        paramArr = nn.utils.parameters_to_vector(self.model.parameters())
        untrainedParamArr = nn.utils.parameters_to_vector(self.untrainedModel.parameters())

        paramNo = len(paramArr)
        shareParamsNo = int(Q * paramNo)

        r = torch.randperm(paramNo).to(self.device)
        paramArr = paramArr[r].to(self.device)
        untrainedParamArr = untrainedParamArr[r].to(self.device)
        paramChanges = (paramArr - untrainedParamArr).detach().to(self.device)

        # Normalising
        if self.needNormalization:
            paramChanges /= self.n * self.epochs

        # Privacy budgets for
        e1 = self.epsilon  # gradient query
        e3 = self.epsilon  # answer
        e2 = e1 * ((2 * shareParamsNo * s) ** (2 / 3))  # threshold

        paramChanges = paramChanges.cpu()
        tau = percentile(abs(paramChanges), Q * 100)
        paramChanges = paramChanges.to(self.device)
        logPrint(f"Raw gradient magnitude: {torch.norm(paramChanges)}")
        # tau = 0.0001
        noisyThreshold = laplace.rvs(scale=(s / e2)) + tau

        queryNoise = laplace.rvs(scale=(2 * shareParamsNo * s / e1), size=paramNo)
        queryNoise = torch.tensor(queryNoise).to(self.device)

        releaseIndex = torch.empty(0).to(self.device)
        while torch.sum(releaseIndex) < shareParamsNo:
            if self.needClip:
                noisyQuery = abs(torch.clamp(paramChanges, -gamma, gamma)) + queryNoise
            else:
                noisyQuery = abs(paramChanges) + queryNoise
            noisyQuery = noisyQuery.to(self.device)
            releaseIndex = (noisyQuery >= noisyThreshold).to(self.device)

        filteredChanges = paramChanges[releaseIndex]

        answerNoise = laplace.rvs(scale=(shareParamsNo * s / e3), size=torch.sum(releaseIndex).cpu())
        answerNoise = torch.tensor(answerNoise).to(self.device)

        logPrint(f"Average queryNoise magnitude: {torch.mean(torch.abs(queryNoise))}")
        logPrint(f"Average answerNoise magnitude: {torch.mean(torch.abs(torch.tensor(answerNoise)))}")
        if self.needClip:
            noisyFilteredChanges = torch.clamp(filteredChanges + answerNoise, -gamma, gamma)
            logPrint(f"Clipped gradient magnitude: {torch.norm(noisyFilteredChanges)}")
        else:
            noisyFilteredChanges = filteredChanges + answerNoise
        noisyFilteredChanges = noisyFilteredChanges.to(self.device)

        # Demoralising the noise
        if self.needNormalization:
            noisyFilteredChanges *= self.n * self.epochs

        logPrint("Broadcast: {}\t"
                 "Trained: {}\t"
                 "Released: {}\t"
                 "answerNoise: {}\t"
                 "ReleasedChange: {}\t"
                 "".format(untrainedParamArr[releaseIndex][0],
                           paramArr[releaseIndex][0],
                           untrainedParamArr[releaseIndex][0] + noisyFilteredChanges[0],
                           answerNoise[0],
                           noisyFilteredChanges[0]))
        # sys.stdout.flush()
        logPrint(f"Noisy gradient magnitude: {torch.norm(noisyFilteredChanges)}")

        paramArr = untrainedParamArr
        paramArr[releaseIndex][:shareParamsNo] += noisyFilteredChanges[:shareParamsNo]
        paramArr = paramArr.to(self.device)
        logPrint(f"Sample parameter values after update: {paramArr[:5]}")
        nn.utils.vector_to_parameters(paramArr, self.model.parameters())
        
        return self.model.to(self.device)


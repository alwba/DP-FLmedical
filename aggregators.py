import copy
import sys
import numpy as np
from scipy.stats import beta
from torch import nn
from logger import logPrint
from threading import Thread
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import csv
import torch


class Aggregator:
    def __init__(self, clients, model, rounds, device, exp_name, useAsyncClients=False):
        self.model = model.to(device)
        self.clients = clients
        self.rounds = rounds

        self.device = device
        self.exp_name = exp_name
        self.useAsyncClients = useAsyncClients

    def trainAndTest(self, testDataset):
        raise Exception("Train method should be override by child class, "
                        "specific to the aggregation strategy.")

    def _shareModelAndTrainOnClients(self):
        if self.useAsyncClients:
            threads = []
            for client in self.clients:
                t = Thread(target=(lambda: self.__shareModelAndTrainOnClient(client)))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
        else:
            for client in self.clients:
                self.__shareModelAndTrainOnClient(client)

    def __shareModelAndTrainOnClient(self, client):
        broadcastModel = copy.deepcopy(self.model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def _retrieveClientModelsDict(self):
        models = dict()
        for client in self.clients:
            # If client blocked return an the unchanged version of the model
            if not client.blocked:
                models[client] = client.retrieveModel()
            else:
                models[client] = client.model
        return models

    def test(self, testDataset):
        dataLoader = DataLoader(testDataset, shuffle=False)
        with torch.no_grad():
            predLabels, testLabels = zip(*[(self.predict(self.model, x), y) for x, y in dataLoader])
        predLabels = torch.tensor(predLabels, dtype=torch.long)
        testLabels = torch.tensor(testLabels, dtype=torch.long)
        # Confusion matrix and normalized confusion matrix
        mconf = confusion_matrix(testLabels, predLabels)

        numOfFeatures = next(iter(dataLoader))[0].shape[1]
        
        protectedAttributesList = []
        if numOfFeatures == 11: # heart dataset
            age_grouping = [28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78]

            protectedAttributesList = [
                {"description": "heart_age", "isBinary": False, "protectedAttributeIndex": 0, "grouping": age_grouping},
                {"description": "heart_gender", "isBinary": True, "protectedAttributeIndex": 1, "unprivilegedValue": 0},
            ]

        elif numOfFeatures == 8: # diabetes
            age_grouping = [16,32,48,64]
            bmi_grouping = [18.5, 25, 30, 40]
            pregnancies_grouping = [0,3,6,9,12,15,18]

            protectedAttributesList = [
                {"description": "diabetes_pregnancies", "isBinary": False, "protectedAttributeIndex": 0, "grouping": pregnancies_grouping},
                {"description": "diabetes_bmi", "isBinary": False, "protectedAttributeIndex": 5, "grouping": bmi_grouping},
                {"description": "diabetes_age", "isBinary": False, "protectedAttributeIndex": 7, "grouping": age_grouping},
            ]

        else:
            print("no suitable fairness measures...") 

        for a in protectedAttributesList:
            protectedAttributes = np.array([x[0][a["protectedAttributeIndex"]].item() for x,_ in dataLoader])

            if a["isBinary"]:
                indices_p = np.where(protectedAttributes != a["unprivilegedValue"])[0]
                indices_u = np.where(protectedAttributes == a["unprivilegedValue"])[0]

                if len(indices_p) == 0 or len(indices_u) == 0:
                    logPrint("no grouping possible")
                    continue

                testLabels_p = testLabels[indices_p]
                testLabels_u = testLabels[indices_u]
                predLabels_p = predLabels[indices_p]
                predLabels_u = predLabels[indices_u]

                mconf_p = confusion_matrix(testLabels_p, predLabels_p)
                mconf_u = confusion_matrix(testLabels_u, predLabels_u)

                logPrint("Fairness measures for", a["description"], "having the unprivileged value of:", a["unprivilegedValue"])
                fairness_measures = self._measureFairness(mconf_p, mconf_u)
                logPrint("----- performance measures privileged group -----")
                performance_measures_p = self._measurePerformance(mconf_p)
                performance_measures_p = {'p_' + key: value for key, value in performance_measures_p.items()}
                logPrint("----- performance measures unprivileged group -----")
                performance_measures_u = self._measurePerformance(mconf_u)
                performance_measures_u = {'u_' + key: value for key, value in performance_measures_u.items()}

                results = {**fairness_measures, **performance_measures_p, **performance_measures_u}

                self._append_to_csv('out/' + self.exp_name + '/' + a["description"] + '.csv', results)
                
            else:
                grouping = a["grouping"]
                for i in range(len(grouping) + 1):
                    filename = ''
                    if i == 0:
                        indices_p = np.where(protectedAttributes >= grouping[i])[0]
                        indices_u = np.where(protectedAttributes < grouping[i])[0]
                        logPrint("Fairness measures for", a["description"], "having a value of <", grouping[i])
                        filename = 'out/' + self.exp_name + '/' + a["description"] + '_x<' + str(grouping[i]) + '.csv'
                    elif i == len(grouping):
                        indices_p = np.where(protectedAttributes < grouping[i-1])[0]
                        indices_u = np.where(protectedAttributes >= grouping[i-1])[0]
                        logPrint("Fairness measures for", a["description"], "having a value >=", grouping[i-1])
                        filename = 'out/' + self.exp_name + '/' + a["description"] + '_x>=' + str(grouping[i-1]) + '.csv'
                    else:
                        indices_p = np.where(np.logical_or(protectedAttributes < grouping[i-1], protectedAttributes >= grouping[i]))[0]
                        indices_u = np.where(np.logical_and(protectedAttributes >= grouping[i-1], protectedAttributes < grouping[i]))[0]
                        logPrint("Fairness measures for", a["description"], "having a value >=", grouping[i-1], "and <", grouping[i])
                        filename = 'out/' + self.exp_name + '/' + a["description"] + '_' + str(grouping[i-1]) + '<=x<' + str(grouping[i]) + '.csv'

                    if len(indices_p) == 0 or len(indices_u) == 0:
                        logPrint("no grouping possible")
                        continue

                    testLabels_p = testLabels[indices_p]
                    testLabels_u = testLabels[indices_u]
                    predLabels_p = predLabels[indices_p]
                    predLabels_u = predLabels[indices_u]

                    mconf_p = confusion_matrix(testLabels_p, predLabels_p, labels=[0,1])
                    mconf_u = confusion_matrix(testLabels_u, predLabels_u, labels=[0,1])

                    fairness_measures = self._measureFairness(mconf_p, mconf_u)
                    logPrint("----- performance measures privileged group -----")
                    performance_measures_p = self._measurePerformance(mconf_p)
                    performance_measures_p = {'p_' + key: value for key, value in performance_measures_p.items()}
                    logPrint("----- performance measures unprivileged group -----")
                    performance_measures_u = self._measurePerformance(mconf_u)
                    performance_measures_u = {'u_' + key: value for key, value in performance_measures_u.items()}

                    results = {**fairness_measures, **performance_measures_p, **performance_measures_u}

                    self._append_to_csv(filename, results)

        logPrint("----- model performance -----")
        model_performance_measures = self._measurePerformance(mconf)
        self._append_to_model_performance_csv('out/' + self.exp_name + '/model_performance.csv', model_performance_measures)
        errors = 1 - 1.0 * mconf.diagonal().sum() / len(testDataset)
        logPrint("Error Rate: ", round(100.0 * errors, 3), "%")
        return errors
    
    def _append_to_csv(self, path, data):
        with open(path, 'a', newline='') as file:
            fieldnames = ['DI_degree', 'EOP_difference', 'EODD_difference', 'SP_difference', 'p_accuracy', 'p_precision', 'p_recall', 'p_f1', 'u_accuracy', 'u_precision', 'u_recall', 'u_f1']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if file.tell() == 0:
                writer.writeheader()

            writer.writerow({key: data.get(key, '') for key in fieldnames})

    def _append_to_model_performance_csv(self, path, data):
        with open(path, 'a', newline='') as file:
            fieldnames = ['accuracy', 'precision', 'recall', 'f1']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if file.tell() == 0:
                writer.writeheader()

            writer.writerow({key: data.get(key, '') for key in fieldnames})
    
    def _measureFairness(self, mconf_p, mconf_u):
        DI_degree = round(self._DI_degree(mconf_p, mconf_u), 4)
        EOP_difference = round(self._EOP_difference(mconf_p, mconf_u), 4)
        EODD_difference = round(self._EODD_difference(mconf_p, mconf_u), 4)
        SP_difference = round(self._SP_difference(mconf_p, mconf_u), 4)

        logPrint("Disparate Impact: ", DI_degree)
        logPrint("EOP difference: ", EOP_difference)
        logPrint("EODD difference: ", EODD_difference)
        logPrint("SP difference: ", SP_difference)

        return {"DI_degree": DI_degree, "EOP_difference": EOP_difference, "EODD_difference": EODD_difference, "SP_difference": SP_difference}

    def _measurePerformance(self, mconf):
        tn, fp, fn, tp = mconf.ravel()

        accuracy = round((tp + tn) / (tn + fp + fn + tp), 4)
        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        f1 = round(2 * (precision * recall) / (precision + recall), 4)

        logPrint("Accuracy: ", accuracy)
        logPrint("Precision: ", precision)
        logPrint("Recall: ", recall)
        logPrint("F1 Score: ", f1)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    def _DI_degree(self, mconf_p, mconf_u):
        # legal threshold for DI is 0.8 or 0.9
        # as far as I understand it it should be more a how many got positive outcome of the whole --> Zafar
        # |1-max(TPR_u/TPR_p, TPR_p/TPR_u)|
        # it is more about the unfavourable outcomes 
        # so maybe |1-(SR_u/SR_p)| --> where SR is the selection rate meaning how meaning got the undesired outcome from the whole
        TN_p, FP_p, FN_p, TP_p = mconf_p.ravel()
        TN_u, FP_u, FN_u, TP_u = mconf_u.ravel()

        PR_p = FP_p + TP_p
        PR_u = FP_u + TP_u
        N_p = TN_p + FP_p + FN_p + TP_p
        N_u = TN_u + FP_u + FN_u + TP_u

        SR_p = PR_p/N_p
        SR_u = PR_u/N_u

        if SR_p == 0 or SR_u == 0:
            return 1

        # expecting SR_p being higher...
        DI_degree = abs(1-SR_u/SR_p)

        return DI_degree

    def _EOP_difference(self, mconf_p, mconf_u):
        # |TPR_p - TPR_u|
        TN_p, FP_p, FN_p, TP_p = mconf_p.ravel()
        TN_u, FP_u, FN_u, TP_u = mconf_u.ravel()

        TPR_p = TP_p/(TP_p + FN_p)
        TPR_u = TP_u/(TP_u + FN_u)

        EOP_difference = abs(TPR_p - TPR_u)

        return EOP_difference

    def _EODD_difference(self, mconf_p, mconf_u):
        # 0.5 * (|TPR_p - TPR_u| + |TNR_p - TNR_u|)
        TN_p, FP_p, FN_p, TP_p = mconf_p.ravel()
        TN_u, FP_u, FN_u, TP_u = mconf_u.ravel()

        TPR_p = TP_p/(TP_p + FN_p)
        TPR_u = TP_u/(TP_u + FN_u)

        TNR_p = TN_p/(TN_p + FP_p)
        TNR_u = TN_u/(TN_u + FP_u)

        EODD_difference = 0.5 * (abs(TPR_p - TPR_u) + abs(TNR_p - TNR_u))
        
        return EODD_difference

    def _SP_difference(self, mconf_p, mconf_u):
        # |((TP_p + FP_p)/N_p) - ((TP_u + FP_u)/N_u)|
        TN_p, FP_p, FN_p, TP_p = mconf_p.ravel()
        TN_u, FP_u, FN_u, TP_u = mconf_u.ravel()
        
        N_p = TN_p + FP_p + FN_p + TP_p
        N_u = TN_u + FP_u + FN_u + FN_u

        SP_difference = abs(((TP_p + FP_p)/N_p) - ((TP_u + FP_u)/N_u))
        
        return SP_difference

    # Function for computing predictions
    def predict(self, net, x):
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)
        return predicted.to(self.device)

    # Function to merge the models
    @staticmethod
    def _mergeModels(mOrig, mDest, alphaOrig, alphaDest):
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                weightedSum = alphaOrig * param1.data \
                              + alphaDest * dictParamsDest[name1].data
                dictParamsDest[name1].data.copy_(weightedSum)


# FEDERATED AVERAGING AGGREGATOR
class FAAggregator(Aggregator):

    def trainAndTest(self, testDataset):
        roundsError = torch.zeros(self.rounds)
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            # Merge models
            comb = 0.0
            for client in self.clients:
                self._mergeModels(models[client].to(self.device), self.model.to(self.device), client.p, comb)
                comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError


# ROBUST AGGREGATION ALGORITHM - computes the median of the clients updates
class COMEDAggregator(Aggregator):

    def trainAndTest(self, testDataset):
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            self.model = self.__medianModels(models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __medianModels(self, models):
        client1 = self.clients[0]
        model = models[client1]
        modelCopy = copy.deepcopy(model)
        params = model.named_parameters()
        for name1, param1 in params:
            m = []
            for client2 in self.clients:
                params2 = models[client2].named_parameters()
                dictParams2 = dict(params2)
                m.append(dictParams2[name1].data.view(-1).to("cpu").numpy())
                # logPrint("Size: ", dictParams2[name1].data.size())
            m = torch.tensor(m)
            med = torch.median(m, dim=0)[0]
            dictParamsm = dict(modelCopy.named_parameters())
            dictParamsm[name1].data.copy_(med.view(dictParamsm[name1].data.size()))
            # logPrint("Median computed, size: ", med.size())
        return modelCopy.to(self.device)


class MKRUMAggregator(Aggregator):

    def trainAndTest(self, testDataset):
        userNo = len(self.clients)
        # Number of Byzantine workers to be tolerated
        f = int((userNo - 3) / 2)
        th = userNo - f - 2
        mk = userNo - f

        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()

            # Compute distances for all users
            scores = torch.zeros(userNo)
            models = self._retrieveClientModelsDict()
            for client in self.clients:
                distances = torch.zeros((userNo, userNo))
                for client2 in self.clients:
                    if client.id != client2.id:
                        distance = self.__computeModelDistance(models[client].to(self.device),
                                                               models[client2].to(self.device))
                        distances[client.id - 1][client2.id - 1] = distance
                dd = distances[client.id - 1][:].sort()[0]
                dd = dd.cumsum(0)
                scores[client.id - 1] = dd[th]

            _, idx = scores.sort()
            selected_users = idx[:mk - 1] + 1
            # logPrint("Selected users: ", selected_users)

            comb = 0.0
            for client in self.clients:
                if client.id in selected_users:
                    self._mergeModels(models[client].to(self.device), self.model.to(self.device), 1 / mk, comb)
                    comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __computeModelDistance(self, mOrig, mDest):
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        d1 = torch.tensor([]).to(self.device)
        d2 = torch.tensor([]).to(self.device)
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
        sim = torch.norm(d1 - d2, p=2)
        return sim


# ADAPTIVE FEDERATED AVERAGING
class AFAAggregator(Aggregator):

    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, useAsyncClients)
        self.xi = 2
        self.deltaXi = 0.5

    def trainAndTest(self, testDataset):
        # List of malicious users blocked
        maliciousBlocked = []
        # List with the iteration where a malicious user was blocked
        maliciousBlockedIt = []
        # List of benign users blocked
        benignBlocked = []
        # List with the iteration where a benign user was blocked
        benignBlockedIt = []

        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):

            logPrint("Round... ", r)

            for client in self.clients:
                broadcastModel = copy.deepcopy(self.model)
                client.updateModel(broadcastModel)
                if not client.blocked:
                    error, pred = client.trainModel()

            models = self._retrieveClientModelsDict()

            badCount = 2
            slack = self.xi
            while badCount != 0:
                pT_epoch = 0.0
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.pEpoch = client.n * client.score
                        pT_epoch = pT_epoch + client.pEpoch

                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.pEpoch = client.pEpoch / pT_epoch

                comb = 0.0
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        self._mergeModels(models[client].to(self.device), self.model.to(self.device), client.pEpoch,
                                          comb)
                        comb = 1.0

                sim = []
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.sim = self.__modelSimilarity(self.model, models[client])
                        sim.append(np.asarray(client.sim.to("cpu")))
                        # logPrint("Similarity user ", u.id, ": ", u.sim)

                sim = np.asarray(sim)

                meanS = np.mean(sim)
                medianS = np.median(sim)
                desvS = np.std(sim)

                if meanS < medianS:
                    th = medianS - slack * desvS
                else:
                    th = medianS + slack * desvS

                slack += self.deltaXi

                badCount = 0
                for client in self.clients:
                    if not client.badUpdate:
                        # Malicious self.clients are below the threshold
                        if meanS < medianS:
                            if client.sim < th:
                                # logPrint("Type1")
                                # logPrint("Bad update from user ", u.id)
                                client.badUpdate = True
                                badCount += 1
                                # Malicious self.clients are above the threshold
                        else:
                            if client.sim > th:
                                client.badUpdate = True
                                badCount += 1

            pT = 0.0
            for client in self.clients:
                if not client.blocked:
                    self.updateUserScore(client)
                    client.blocked = self.checkBlockedUser(client.alpha, client.beta)
                    if client.blocked:
                        logPrint("USER ", client.id, " BLOCKED!!!")
                        client.p = 0
                        if client.byz:
                            maliciousBlocked.append(client.id)
                            maliciousBlockedIt.append(r)
                        else:
                            benignBlocked.append(client.id)
                            benignBlockedIt.append(r)
                    else:
                        client.p = client.n * client.score
                        pT = pT + client.p

            for client in self.clients:
                client.p = client.p / pT
                # logPrint("Weight user", u.id, ": ", round(u.p,3))

            # Update model with the updated scores
            pT_epoch = 0.0
            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.n * client.score
                    pT_epoch = pT_epoch + client.pEpoch

            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.pEpoch / pT_epoch
            # logPrint("Updated scores:{}".format([client.pEpoch for client in self.clients]))
            comb = 0.0
            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    self._mergeModels(models[client].to(self.device), self.model.to(self.device), client.pEpoch, comb)
                    comb = 1.0

            # Reset badUpdate variable
            for client in self.clients:
                if not client.blocked:
                    client.badUpdate = False

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __modelSimilarity(self, mOrig, mDest):
        cos = nn.CosineSimilarity(0)

        d2 = torch.tensor([]).to(self.device)
        d1 = torch.tensor([]).to(self.device)

        paramsOrig = mOrig.named_parameters()
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)

        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
                # d2 = param1.data
                # sim = cos(d1.view(-1),d2.view(-1))
                # logPrint(name1,param1.size())
                # logPrint("Similarity: ",sim)
        sim = cos(d1, d2)
        return sim

    @staticmethod
    def checkBlockedUser(a, b, th=0.95):
        # return beta.cdf(0.5, a, b) > th
        s = beta.cdf(0.5, a, b)
        blocked = False
        if s > th:
            blocked = True
        return blocked

    @staticmethod
    def updateUserScore(client):
        if client.badUpdate:
            client.beta += 1
        else:
            client.alpha += 1
        client.score = client.alpha / client.beta

    @staticmethod
    def notBlockedNorBadUpdate(client):
        return client.blocked == False | client.badUpdate == False


def allAggregators():
    return Aggregator.__subclasses__()


# FederatedAveraging and Adaptive Federated Averaging
def FAandAFA():
    return [FAAggregator, AFAAggregator]

# just FedAvg
def FA():
    return [FAAggregator]

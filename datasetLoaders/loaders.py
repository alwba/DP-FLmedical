import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from logger import logPrint


class DatasetInterface(Dataset):

    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise Exception("Method should be implemented in subclass.")

    def getInputSize(self):
        raise Exception("Method should be implemented by subclasses where "
                        "models requires input size update (based on dataset).")

    def zeroLabels(self):
        self.labels = torch.zeros(len(self.labels), dtype=torch.long)


class DatasetLoader:
    """Parent class used for specifying the data loading workflow """

    def getDatasets(self, percUsers, labels, size=(None, None)):
        raise Exception("LoadData method should be override by child class, "
                        "specific to the loaded dataset strategy.")

    @staticmethod
    def _filterDataByLabel(labels, trainDataframe, testDataframe):
        trainDataframe = trainDataframe[trainDataframe['labels'].isin(labels)]
        testDataframe = testDataframe[testDataframe['labels'].isin(labels)]
        return trainDataframe, testDataframe

    @staticmethod
    def _splitTrainDataIntoClientDatasets(percUsers, trainDataframe, DatasetType):
        DatasetLoader._setRandomSeeds()
        percUsers = percUsers / percUsers.sum()

        dataSplitCount = (percUsers * len(trainDataframe)).floor().numpy()
        _, *dataSplitIndex = [int(sum(dataSplitCount[range(i)])) for i in range(len(dataSplitCount))]

        trainDataframes = np.split(trainDataframe, indices_or_sections=dataSplitIndex)
        clientDatasets = [DatasetType(clientDataframe.reset_index(drop=True))
                          for clientDataframe in trainDataframes]
        return clientDatasets

    @staticmethod
    def _setRandomSeeds(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # When anonymizing the clients' datasets using  _anonymizeClientDatasets the function passed as
    #  parameter should take as parameter the cn.protect object and set ds specific generalisations
    @staticmethod
    def _anonymizeClientDatasets(clientDatasets, columns, k, quasiIds, setHierarchiesMethod):

        datasetClass = clientDatasets[0].__class__

        resultDataframes = []
        clientSyntacticMappings = []

        dataframes = [pd.DataFrame(list(ds.dataframe['data']), columns=columns) for ds in clientDatasets]
        for dataframe in dataframes:
            anonIndex = dataframe.groupby(quasiIds)[dataframe.columns[0]].transform('size') >= k

            anonDataframe = dataframe[anonIndex]
            needProtectDataframe = dataframe[~anonIndex]

            # Might want to ss those for the report:
            # print(anonDataframe)
            # print(needProtectDataframe)

            protect = Protect(needProtectDataframe, KAnonymity(k))
            protect.quality_model = quality.Loss()
            # protect.quality_model = quality.Classification()
            protect.suppression = 0

            for qid in quasiIds:
                protect.itypes[qid] = 'quasi'

            setHierarchiesMethod(protect)

            protectedDataframe = protect.protect()
            mappings = protectedDataframe[quasiIds].drop_duplicates().to_dict('records')
            clientSyntacticMappings.append(mappings)
            protectedDataframe = pd.get_dummies(protectedDataframe)

            resultDataframe = pd.concat([anonDataframe, protectedDataframe]).fillna(0).sort_index()
            resultDataframes.append(resultDataframe)

        # All clients datasets should have same columns
        allColumns = set().union(*[df.columns.values for df in resultDataframes])
        for resultDataframe in resultDataframes:
            for col in allColumns - set(resultDataframe.columns.values):
                resultDataframe[col] = 0

        # Create new datasets by adding the labels to
        anonClientDatasets = []
        for resultDataframe, initialDataset in zip(resultDataframes, clientDatasets):
            labels = initialDataset.dataframe['labels'].values
            labeledDataframe = pd.DataFrame(zip(resultDataframe.values, labels))
            labeledDataframe.columns = ['data', 'labels']
            anonClientDatasets.append(datasetClass(labeledDataframe))

        return anonClientDatasets, clientSyntacticMappings, allColumns

    def _anonymizeTestDataset(self, testDataset, clientSyntacticMappings, columns, generalizedColumns):

        datasetClass = testDataset.__class__
        dataframe = pd.DataFrame(list(testDataset.dataframe['data']), columns=columns)

        domainsSize = dict()
        quasiIds = clientSyntacticMappings[0][0].keys()
        for quasiId in quasiIds:
            domainsSize[quasiId] = dataframe[quasiId].max() - dataframe[quasiId].min()

        generalisedDataframe = pd.DataFrame(dataframe)
        ungeneralisedIndex = []
        for i in range(len(dataframe)):
            legitMappings = []
            for clientMappings in clientSyntacticMappings:
                legitMappings += [mapping for mapping in clientMappings
                                  if self.__legitMapping(dataframe.iloc[i], mapping)]
            if legitMappings:
                # leastGeneralMapping = reduce(self.__leastGeneral, legitMappings)
                leastGeneralMapping = legitMappings[0]
                for legitMapping in legitMappings[1:]:
                    leastGeneralMapping = self.__leastGeneral(leastGeneralMapping, legitMapping, domainsSize)

                for col in leastGeneralMapping:
                    generalisedDataframe[col][i] = leastGeneralMapping[col]
            else:
                ungeneralisedIndex.append(i)
                generalisedDataframe = generalisedDataframe.drop(i)

        generalisedDataframe = pd.get_dummies(generalisedDataframe)
        ungeneralisedDataframe = dataframe.iloc[ungeneralisedIndex]

        resultDataframe = pd.concat([ungeneralisedDataframe, generalisedDataframe]).fillna(0).sort_index()
        for col in generalizedColumns - set(resultDataframe.columns.values):
            resultDataframe[col] = 0

        labels = testDataset.dataframe['labels'].values
        labeledDataframe = pd.DataFrame(zip(resultDataframe.values, labels))
        labeledDataframe.columns = ['data', 'labels']

        return datasetClass(labeledDataframe)

    @staticmethod
    def __leastGeneral(map1, map2, domainSize):
        map1Generality = map2Generality = 0
        for col in map1:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r'\d+.\d+', map1[col]), dtype=np.float)
                map1Generality += (interval[1] - interval[0]) / domainSize[col]

        for col in map2:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r'\d+.\d+', map2[col]), dtype=np.float)
                map2Generality += (interval[1] - interval[0]) / domainSize[col]

        return map1 if map1Generality <= map2Generality else map2

    @staticmethod
    def __legitMapping(entry, mapping):
        for col in mapping:
            if not isinstance(mapping[col], str):
                if entry[col] != mapping[col]:
                    return False
            else:
                interval = np.array(re.findall(r'\d+.\d+', mapping[col]), dtype=np.float)
                if interval[0] < entry[col] or entry[col] >= interval[1]:
                    return False
        return True

class DatasetLoaderDiabetes(DatasetLoader):

    def __init__(self, requiresAnonymization=False):
        self.requireDatasetAnonymization = requiresAnonymization

        # Parameters required by k-anonymity enforcement
        self.k = 4
        self.quasiIds = ['Pregnancies', 'Age']

    def getDatasets(self, percUsers, labels, size=None):
        logPrint("Loading Diabetes data...")
        self._setRandomSeeds()
        trainDataframe, testDataframe, columns = self.__loadDiabetesData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, trainDataframe, testDataframe)
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.DiabetesDataset)
        testDataset = self.DiabetesDataset(testDataframe)

        return clientDatasets, testDataset

    @staticmethod
    def __loadDiabetesData(dataBinning=False):
        data = pd.read_csv('data/Diabetes/diabetes.csv')
        # Shuffle
        data = data.sample(frac=1).reset_index(drop=True)

        # Handling Missing Data¶
        data['BMI'] = data.BMI.mask(data.BMI == 0, (data['BMI'].mean(skipna=True)))
        data['BloodPressure'] = data.BloodPressure.mask(data.BloodPressure == 0,
                                                        (data['BloodPressure'].mean(skipna=True)))
        data['Glucose'] = data.Glucose.mask(data.Glucose == 0, (data['Glucose'].mean(skipna=True)))

        # data = data.drop(['Insulin'], axis=1)
        # data = data.drop(['SkinThickness'], axis=1)
        # data = data.drop(['DiabetesPedigreeFunction'], axis=1)

        labels = data['Outcome']
        data = data.drop(['Outcome'], axis=1)

        if dataBinning:
            data['Age'] = data['Age'].astype(int)
            data.loc[data['Age'] <= 16, 'Age'] = 0
            data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
            data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
            data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
            data.loc[data['Age'] > 64, 'Age'] = 4

            data['Glucose'] = data['Glucose'].astype(int)
            data.loc[data['Glucose'] <= 80, 'Glucose'] = 0
            data.loc[(data['Glucose'] > 80) & (data['Glucose'] <= 100), 'Glucose'] = 1
            data.loc[(data['Glucose'] > 100) & (data['Glucose'] <= 125), 'Glucose'] = 2
            data.loc[(data['Glucose'] > 125) & (data['Glucose'] <= 150), 'Glucose'] = 3
            data.loc[data['Glucose'] > 150, 'Glucose'] = 4

            data['BloodPressure'] = data['BloodPressure'].astype(int)
            data.loc[data['BloodPressure'] <= 50, 'BloodPressure'] = 0
            data.loc[(data['BloodPressure'] > 50) & (data['BloodPressure'] <= 65), 'BloodPressure'] = 1
            data.loc[(data['BloodPressure'] > 65) & (data['BloodPressure'] <= 80), 'BloodPressure'] = 2
            data.loc[(data['BloodPressure'] > 80) & (data['BloodPressure'] <= 100), 'BloodPressure'] = 3
            data.loc[data['BloodPressure'] > 100, 'BloodPressure'] = 4

        xTrain = data.head(int(len(data) * .8)).values
        xTest = data.tail(int(len(data) * .2)).values
        yTrain = labels.head(int(len(data) * .8)).values
        yTest = labels.tail(int(len(data) * .2)).values

        trainDataframe = pd.DataFrame(zip(xTrain, yTrain))
        testDataframe = pd.DataFrame(zip(xTest, yTest))
        trainDataframe.columns = testDataframe.columns = ['data', 'labels']

        return trainDataframe, testDataframe, data.columns

    # @staticmethod
    # def __setHierarchies(protect):
    #     protect.hierarchies.Age = OrderHierarchy('interval', 1, 5, 2, 2, 2)
    #     protect.hierarchies.Pregnancies = OrderHierarchy('interval', 1, 2, 2, 2, 2)

    class DiabetesDataset(DatasetInterface):

        def __init__(self, dataframe):
            self.dataframe = dataframe
            self.data = torch.stack([torch.from_numpy(data) for data in dataframe['data'].values], dim=0).float()
            super().__init__(dataframe['labels'].values)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def getInputSize(self):
            return len(self.dataframe['data'][0])


class DatasetLoaderHeartDisease(DatasetLoader):

    def __init__(self, requiresAnonymization=False):
        self.requireDatasetAnonymization = requiresAnonymization
        # Parameters required by k-anonymity enforcement
        self.k = 2
        self.quasiIds = ['age', 'sex']

    def getDatasets(self, percUsers, labels, size=None):
        logPrint("Loading Heart Disease data...")
        self._setRandomSeeds()
        trainDataframe, testDataframe, columns = self.__loadHeartDiseaseData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, trainDataframe, testDataframe)
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.HeartDiseaseDataset)
        testDataset = self.HeartDiseaseDataset(testDataframe)

        return clientDatasets, testDataset

    @staticmethod
    def __loadHeartDiseaseData():
        data = pd.read_csv('data/HeartDisease/heart.csv')
        # Shuffle
        data = data.sample(frac=1).reset_index(drop=True)

        labels = data['Outcome']
        data = data.drop(['Outcome'], axis=1)

        # mappings
        sex_mapping = {'M': 1, 'F': 0}
        cp_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        restecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
        exang_mapping = {'Y': 1, 'N':0}
        slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

        data['Sex'] = data['Sex'].map(sex_mapping)
        data['ChestPainType'] = data['ChestPainType'].map(cp_mapping)
        data['RestingECG'] = data['RestingECG'].map(restecg_mapping)
        data['ExerciseAngina'] = data['ExerciseAngina'].map(exang_mapping)
        data['ST_Slope'] = data['ST_Slope'].map(slope_mapping)

        # train test split
        xTrain = data.head(int(len(data) * .8)).values
        xTest = data.tail(int(len(data) * .2)).values
        yTrain = labels.head(int(len(data) * .8)).values
        yTest = labels.tail(int(len(data) * .2)).values

        trainDataframe = pd.DataFrame(zip(xTrain, yTrain))
        testDataframe = pd.DataFrame(zip(xTest, yTest))
        trainDataframe.columns = testDataframe.columns = ['data', 'labels']

        return trainDataframe, testDataframe, data.columns

    class HeartDiseaseDataset(DatasetInterface):

        def __init__(self, dataframe):
            self.dataframe = dataframe
            self.data = torch.stack([torch.from_numpy(data) for data in dataframe['data'].values], dim=0).float()
            super().__init__(dataframe['labels'].values)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def getInputSize(self):
            return len(self.dataframe['data'][0])

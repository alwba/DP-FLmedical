import torch
import aggregators as agg


class DefaultExperimentConfiguration:
    def __init__(self):
        # DEFAULT PARAMETERS
        self.name = None

        # Federated learning parameters
        self.rounds = 35  # Total number of training rounds
        self.epochs = 10  # Epochs num locally run by clients before sending back the model update
        self.batchSize = 200  # Local training  batch size
        self.learningRate = 0.1
        self.Loss = torch.nn.CrossEntropyLoss
        self.Optimizer = torch.optim.SGD

        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize = (None, None)

        # Clients setup
        self.percUsers = torch.tensor([0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1])  # Client data partition
        self.labels = torch.tensor(range(10))  # Considered dataset labels
        self.faulty = []  # List of noisy clients
        self.malicious = []  # List of (malicious) clients with flipped labels

        # AFA Parameters:
        self.alpha = 3
        self.beta = 3

        # Client privacy preserving module setup
        self.privacyPreserve = False  # if None, run with AND without DP
        self.releaseProportion = 0.1
        self.epsilon = 1
        self.delta = 1e-5
        self.needClip = False
        self.clipValue = 0.001
        self.needNormalization = False

        # Anonymization of datasets for k-anonymity
        self.requireDatasetAnonymization = False

        self.aggregators = agg.allAggregators()  # Aggregation strategies

        self.plotResults = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _getHeartBaselinePerformanceMeasures():
    clients = [1,3,5,10]

    performance_values = {}
    for c in clients:
        df = pd.read_csv('out/heartDisease/heartDisease_baseline_' + str(c) + '_clients/model_performance.csv')
        performance_values[c] = df.iloc[-1].tolist()

    return performance_values

def _getDiabetesBaselinePerformanceMeasures():
    clients = [1,3,5,10]

    performance_values = {}
    for c in clients:
        df = pd.read_csv('out/diabetes/diabetes_baseline_' + str(c) + '_clients/model_performance.csv')
        performance_values[c] = df.iloc[-1].tolist()
    
    return performance_values


def plotBaselinePerformanceMeasures(performance_values):
    clients = [1,3,5,10]
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    x = np.arange(len(clients))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(performance_metrics):
        values = [performance_values[client][i] for client in clients]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Metrics by Number of Clients')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clients)
    ax.legend()

    fig.tight_layout

    # TODO add title + path
    plt.savefig('out/plots/baseline_performance.png')


def _getFairnessValues(fileName):
    epsilons = [0.0001, 0.1, 1]
    clients = [1,3,5]

    fairness_values_averaged = {}
    for e in epsilons:
        fairness_values = {}
        for c in clients:
            # df = pd.read_csv('out/heartDisease/heartDisease_DP_' + str(c) + '_clients_e=' + str(e) + '/' + fileName)
            df = pd.read_csv('out/diabetes/test_diabetes_DP_' + str(c) + '_clients_e=' + str(e) + '/' + fileName)
            fairness_values[c] = df.iloc[-1].tolist()[:4]
       
        num_lists = len(fairness_values)
        list_length = len(next(iter(fairness_values.values())))
        print(e)
        print(fairness_values)
        fairness_values_averaged[e] = [sum(fairness_values[c][i] for c in fairness_values) / num_lists for i in range(list_length)]
    print('fairness values averaged')
    print(fairness_values_averaged)
    return fairness_values_averaged

def plotFairnessValues(fileName):
    fairnes_values = _getFairnessValues(fileName=fileName)

    epsilons = list(fairnes_values.keys())
    fairness_metrics = ['DI_degree','EOP_difference','EODD_difference','SP_difference']

    x = np.arange(len(fairness_metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, epsilon in enumerate(epsilons):
        values = fairnes_values[epsilon]
        ax.bar(x + i*width, values, width, label=f'Epsilon {epsilon}')

    ax.set_xlabel('Fairness Metrics')
    ax.set_ylabel('Discrimination Level')
    ax.set_title('Fairness Metrics by Epsilon Value')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(fairness_metrics)
    ax.legend()

    plt.show()
    # TODO save + add title


plotFairnessValues("diabetes_pregnancies_6<=x<9.csv")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _getHeartBaselinePerformanceMeasures():
    clients = [1,3,5,10]

    performance_values = {}
    for c in clients:
        df = pd.read_csv('out/heartDisease/heartDisease_baseline_' + str(c) + '_clients/model_performance.csv')
        # extract last row
        performance_values[c] = df.iloc[-1].tolist()

    return performance_values

def _getDiabetesBaselinePerformanceMeasures():
    clients = [1,3,5,10]

    performance_values = {}
    for c in clients:
        df = pd.read_csv('out/diabetes/diabetes_baseline_' + str(c) + '_clients/model_performance.csv')
        # extract last row
        performance_values[c] = df.iloc[-1].tolist()
    
    return performance_values

def _getHeartBaselineFairnessMeasuresGender():
    clients = [1,3,5,10]

    performance_values = {}
    for c in clients:
        df = pd.read_csv('out/heartDisease/heartDisease_baseline_' + str(c) + '_clients/heart_gender.csv')
        # extract last row
        performance_values[c] = df.iloc[-1].tolist()
    
    return performance_values

def plotBaselinePerformanceMeasures(performance_values):
    clients = [1,3,5,10]
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # Colors for each metric
    colors = ['blue', 'green', 'red', 'orange']

    # Plotting
    x = np.arange(len(clients))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(performance_metrics):
        values = [performance_values[client][i] for client in clients]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Metrics by Number of Clients')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clients)
    ax.legend()

    fig.tight_layout()

    plt.show()
    # TODO implement saving the plot!
    # TODO add title

def plotHeartGenderBaselineFairnessMeasures():
    performance_values = _getHeartBaselineFairnessMeasuresGender()
    print(performance_values)

    clients = [1,3,5,10]
    performance_metrics = ['DI_degree','EOP_difference','EODD_difference','SP_difference'] #,'p_accuracy','p_precision','p_recall','p_f1','u_accuracy','u_precision','u_recall','u_f1']
    # Colors for each metric
    colors = ['blue', 'green', 'red', 'orange']

    # Plotting
    x = np.arange(len(clients))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(performance_metrics):
        values = [performance_values[client][i] for client in clients]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Discrimination level')
    ax.set_title('Fairness Metrics by Number of Clients for gender in heart disease')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clients)
    ax.legend()

    fig.tight_layout()

    plt.show()
    # TODO implement saving the plot!
    # TODO add title

def plotFairnessMetrics():
    # TODO collect necessary data
    fairness_values = []

    epsilons = [0.001, 0.1, 1]
    fairness_metrics = ['DI_degree','EOP_difference','EODD_difference','SP_difference']

    colors = ['blue', 'green', 'red', 'orange']

    x = np.arange(len(epsilons))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(fairness_metrics):
        values = [fairness_values[e][i] for e in epsilons]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Discrimination level')
    # TODO adjust title
    ax.set_title('Fairness Metrics')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(epsilons)
    ax.legend()

    fig.tight_layout()

    plt.show()
    # TODO save plot


# performance_values = _getHeartBaselinePerformanceMeasures()
# plotBaselinePerformanceMeasures(performance_values)
# # --> best results for 3 clients
# performance_values = _getDiabetesBaselinePerformanceMeasures()
# plotBaselinePerformanceMeasures(performance_values)
# --> same here


##### not needed for baseline
plotHeartGenderBaselineFairnessMeasures()
# --> fairness wise it looks better for 5 clients for gender heart
# this can be done for all files --> just the title should be changed

# but then we also need a comparison for the performance measures for privileged and unprivileged group
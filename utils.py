import os
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


def make_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_empty = not os.path.exists(path) or os.stat(path).st_size == 0

    data_out = pd.DataFrame({'time_roi_1': [data[0][0]], 'sum_features_roi_1': [data[0][1]],
                             'result_roi_1': [data[0][2]], 'win_roi_1': [data[0][3]], 'time_roi_2': [data[1][0]],
                             'sum_features_roi_2': [data[1][1]], 'result_roi_2': [data[1][2]], 'win_roi_2': [data[1][3]]
                             })
    if file_empty:
        data_out.to_csv(path, mode='w', index=False)
    else:
        data_out.to_csv(path, mode='a', header=False, index=False)


def make_plot():
    plt.figure(figsize=(10, 6))

    data = pd.read_csv('./output/audit_data.csv')

    data['result_roi_1'] = data['result_roi_1'].astype(bool)

    plt.plot(data['time_roi_1'], data['win_roi_2'], color='blue', label='Sum Features ROI 1')
    plt.plot(data['time_roi_1'], data['win_roi_1'], color='green', label='Sum Features ROI 2')

    for result, color in zip([False, True], ['yellow', 'red']):
        plt.scatter(data[data['result_roi_1'] == result]['time_roi_1'],
                    data[data['result_roi_1'] == result]['win_roi_1'],
                    c=color, label=f'Result ROI 1: {result}')

    for result, color in zip([False, True], ['yellow', 'red']):
        plt.scatter(data[data['result_roi_2'] == result]['time_roi_1'],
                    data[data['result_roi_2'] == result]['win_roi_2'],
                    c=color, label=f'Result ROI 2: {result}')

    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Sum Features')
    plt.title('Sum Features ROI 1 and ROI 2 over Time')

    plt.show()


#make_plot()


def make_histo():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Read the data from the CSV file
    data = pd.read_csv('./output/audit_data2.csv')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram for sum_features_roi_1
    axs[0].hist(data['sum_features_roi_1'], bins=50, color='blue', alpha=0.7)
    axs[0].set_title('Sum Features ROI 1')
    axs[0].set_xlabel('Sum Features ROI 1')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    # Histogram for sum_features_roi_2
    axs[1].hist(data['sum_features_roi_2'], bins=50, color='red', alpha=0.7)
    axs[1].set_title('Sum Features ROI 2')
    axs[1].set_xlabel('Sum Features ROI 2')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# make_histo()




def plot_kde(data, ax, title):
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(data.values.reshape(-1, 1))
    x = np.linspace(data.min(), data.max(), 1000)
    log_dens = kde.score_samples(x.reshape(-1, 1))
    ax.plot(x, np.exp(log_dens), label=title)
    ax.set_title(f'Probability Distribution for {title}')
    ax.legend()
    return kde


def prob_dist():
    data = pd.read_csv('./output/audit_data2.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    mean1 = data['sum_features_roi_1'].mean()
    std1 = data['sum_features_roi_1'].std()
    axes[0].text(0.05, 0.95, f"Mean: {mean1:.2f}\nStd: {std1:.2f}", transform=axes[0].transAxes)

    mean2 = data['sum_features_roi_2'].mean()
    std2 = data['sum_features_roi_2'].std()
    axes[1].text(0.05, 0.95, f"Mean: {mean2:.2f}\nStd: {std2:.2f}", transform=axes[1].transAxes)

    plt.tight_layout()
    print('STD1:', std1, 'Mean1:', mean1)
    print('STD2:', std2, 'Mean2:', mean2)


def prob_dist2():
    data = pd.read_csv('./output/audit_data2.csv')
    scaler = MinMaxScaler()
    data[['sum_features_roi_1', 'sum_features_roi_2']] = scaler.fit_transform(
        data[['sum_features_roi_1', 'sum_features_roi_2']])

    # Plot the probability distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data['sum_features_roi_1'], label='Sum Features ROI 1', color='blue', fill=True)
    sns.kdeplot(data['sum_features_roi_2'], label='Sum Features ROI 2', color='red', fill=True)

    plt.title('Probability Distribution of Sum Features')
    plt.xlabel('Normalized Sum Features')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def calc_mean_and_std2(data):
    stds = {}

    if data[0] == 'roi_1':
        stds['roi_1'] = np.std(data[1])
    elif data[0] == 'roi_2':
        stds['roi_2'] = np.std(data[1])

    return stds


def calc_mean_and_std(data):
    stds = []

    if data[0] == 'roi_1':
        stds.append(np.std(data[1], ddof=1))
    elif data[0] == 'roi_2':
        stds.append(np.std(data[1], ddof=1))
    return stds[-1]

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_empty = not os.path.exists(path) or os.stat(path).st_size == 0

    data_out = pd.DataFrame({'time_roi_1': [data[0][0]], 'sum_features_roi_1': [data[0][1]],
                             'result_roi_1': [data[0][2]], 'time_roi_2': [data[1][0]],
                             'sum_features_roi_2': [data[1][1]], 'result_roi_2': [data[1][2]]
                             })
    if file_empty:
        data_out.to_csv(path, mode='w', index=False)
    else:
        data_out.to_csv(path, mode='a', header=False, index=False)


def make_plot():
    plt.figure(figsize=(10, 6))

    data = pd.read_csv('./output/audit_data.csv')

    data['result_roi_1'] = data['result_roi_1'].astype(bool)

    plt.plot(data['time_roi_1'], data['sum_features_roi_1'], color='blue', label='Sum Features ROI 1')
    plt.plot(data['time_roi_1'], data['sum_features_roi_2'], color='green', label='Sum Features ROI 2')

    for result, color in zip([False, True], ['yellow', 'red']):
        plt.scatter(data[data['result_roi_1'] == result]['time_roi_1'],
                    data[data['result_roi_1'] == result]['sum_features_roi_1'],
                    c=color, label=f'Result ROI 1: {result}')

    for result, color in zip([False, True], ['yellow', 'red']):
        plt.scatter(data[data['result_roi_2'] == result]['time_roi_1'],
                    data[data['result_roi_2'] == result]['sum_features_roi_2'],
                    c=color, label=f'Result ROI 1: {result}')


    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Sum Features')
    plt.title('Sum Features ROI 1 and ROI 2 over Time')

    plt.show()


#make_plot()

def make_histo(deque_roi):
    flattened_data = np.concatenate(deque_roi)

    normalized_data = (flattened_data - np.min(flattened_data)) / (np.max(flattened_data) - np.min(flattened_data))
    print(normalized_data)
    # # Plot histogram with normalized data
    # plt.hist(normalized_data, bins=50, alpha=0.7)
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Normalized Histogram of Deque Data')
    # plt.grid(True)
    # plt.show()

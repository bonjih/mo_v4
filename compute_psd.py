import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from hurst import compute_Hc


def compute_psd(df):
    psd_values = []
    psd_list = []

    df = pd.DataFrame(df, columns=['X', 'y'])

    X = df['X']
    y = df['y']

    series = pd.Series(y)

    df.set_index(X, inplace=True)
    df_sum = df.dropna()

    # Filter DataFrame based on 'y' values
    df_filtered = df_sum[df_sum['y'].isin(y)]

    # Normalize the data
    top = np.max(df_filtered['y'])
    bottom = np.min(df_filtered['y'])
    mid = np.average(df_filtered['y'])
    normSamples = (df_filtered['y'] - mid)
    normSamples /= top - bottom

    # Normalize the data
    values = series.values.reshape((-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)

    # get PSD
    pps_fft = sp.fftpack.fft(normalized)
    pps_psd = np.abs(pps_fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(pps_psd))

    # Find the peak frequency, only the positive frequencies
    i = np.where(fftfreq > 0)
    freqs = fftfreq[i]

    # Check for zero or negative PSD values before taking the logarithm
    positive_indices = np.where(pps_psd[i] > 0)[0]  # Get the indices as a 1D array
    freqs_positive = freqs[positive_indices]
    psd_values_positive = 10 * np.log10(pps_psd[i][positive_indices])

    for j, k in zip(list(freqs_positive), list(psd_values_positive)):
        psd_values.append([j, k])
        output = [item[1][0] for item in psd_values]
        psd_val = [', '.join([f'{value:.8f}' for value in output])]
        psd_list.append(psd_val)

    if not psd_values:
        return None

    median = np.median([item[1][0] for item in psd_values])
    return median



import numpy as np
import numba
from numba import cuda
import math
import matplotlib.pyplot as plt
import os

def plot_scalars_old(df, N, D, figsize, block=True):
    df['e'] = df['u'] + df['k']  # Total energy
    df['Tkin'] = 2 * df['k'] / D / (N - 1)
    df['Tconf'] = df['fsq'] / df['lap']
    df['press'] = 2 * df['k'] / D / (N - 1) * N / df['vol'] + df['w'] / df['vol']
    df['du'] = df['u'] - np.mean(df['u'])
    df['de'] = df['e'] - np.mean(df['e'])
    df['dw'] = df['w'] - np.mean(df['w'])

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].plot(df['t'].values, df['du'].values / N, '.-', label=f"du/N, var(u)/N={np.var(df['u']) / N:.4}")
    axs[0, 0].plot(df['t'].values, df['de'].values / N, '-', label=f"de/N, var(e)/N={np.var(df['e']) / N:.4}")
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].legend()

    axs[0, 1].plot(df['t'].values, df['Tconf'].values, '.-', label=f"Tconf, mean={np.mean(df['Tconf']):.3f}")
    axs[0, 1].plot(df['t'].values, df['Tkin'].values, '.-', label=f"Tkin, mean={np.mean(df['Tkin']):.3f}")
    if 'Ttarget' in df.columns:
        axs[0, 1].plot(df['t'].values, df['Ttarget'].values, 'k--', linewidth=3,
                       label=f"Ttarget,  mean={np.mean(df['Ttarget']):.3f}")
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Temperature')
    axs[0, 1].legend()

    axs[1, 0].plot(df['t'].values, df['press'].values, '.-', label=f"press, mean={np.mean(df['press']):.3f}")
    if 'Ptarget' in df.columns:
        axs[1, 0].plot(df['t'].values, df['Ptarget'].values, 'k--', linewidth=3,
                       label=f"Ptarget,  mean={np.mean(df['Ptarget']):.3f}")

    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure')
    axs[1, 0].legend()

    ramp = False
    if 'Ttarget' in df.columns:  # Is it a ramp?
        if np.std(df['Ttarget']) > 0.01 * np.mean(df['Ttarget']):
            ramp = True
            axs[1, 1].plot(df['Ttarget'].values, df['u'].values / N, '.-')
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Potenital energy per particle')

    if ramp == False:
        R = np.dot(df['dw'], df['du']) / (np.dot(df['dw'], df['dw']) * np.dot(df['du'], df['du'])) ** 0.5
        Gamma = np.dot(df['dw'], df['du']) / (np.dot(df['du'], df['du']))

        axs[1, 1].plot(df['u'].values / N, df['w'].values / N, '.', label=f"R = {R:.3}")
        axs[1, 1].plot(sorted(df['u'].values / N), sorted(df['du'].values / N * Gamma + np.mean(df['w'].values / N)),
                       'r--', label=f"Gamma = {Gamma:.3}")
        axs[1, 1].set_xlabel('U/N')
        axs[1, 1].set_ylabel('W/N')
        axs[1, 1].legend()

    if __name__ == "__main__":
        plt.show(block=block)

    return


def plot_scalars(df, N, D, figsize, block=True):
    df['E'] = df['U'] + df['K']  # Total energy
    df['Tkin'] = 2 * df['K'] / D / (N - 1)
    df['Tconf'] = df['Fsq'] / df['lapU']
    df['press'] = 2 * df['K'] / D / (N - 1) * N / df['Vol'] + df['W'] / df['Vol']
    df['dU'] = df['U'] - np.mean(df['U'])
    df['dE'] = df['E'] - np.mean(df['E'])
    df['dW'] = df['W'] - np.mean(df['W'])

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].plot(df['t'].values, df['dU'].values / N, '.-', label=f"dU/N, var(U)/N={np.var(df['U']) / N:.4}")
    axs[0, 0].plot(df['t'].values, df['dE'].values / N, '-', label=f"dE/N, var(E)/N={np.var(df['E']) / N:.4}")
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].legend()

    axs[0, 1].plot(df['t'].values, df['Tconf'].values, '.-', label=f"Tconf, mean={np.mean(df['Tconf']):.3f}")
    axs[0, 1].plot(df['t'].values, df['Tkin'].values, '.-', label=f"Tkin, mean={np.mean(df['Tkin']):.3f}")
    if 'Ttarget' in df.columns:
        axs[0, 1].plot(df['t'].values, df['Ttarget'].values, 'k--', linewidth=3,
                       label=f"Ttarget,  mean={np.mean(df['Ttarget']):.3f}")
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Temperature')
    axs[0, 1].legend()

    axs[1, 0].plot(df['t'].values, df['press'].values, '.-', label=f"press, mean={np.mean(df['press']):.3f}")
    if 'Ptarget' in df.columns:
        axs[1, 0].plot(df['t'].values, df['Ptarget'].values, 'k--', linewidth=3,
                       label=f"Ptarget,  mean={np.mean(df['Ptarget']):.3f}")

    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure')
    axs[1, 0].legend()

    ramp = False
    if 'Ttarget' in df.columns:  #
        if np.std(df['Ttarget']) > 0.01 * np.mean(df['Ttarget']):
            ramp = True
            axs[1, 1].plot(df['Ttarget'].values, df['U'].values / N, '.-')
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Potenital energy per particle')

    if ramp == False:
        R = np.dot(df['dW'], df['dU']) / (np.dot(df['dW'], df['dW']) * np.dot(df['dU'], df['dU'])) ** 0.5
        Gamma = np.dot(df['dW'], df['dU']) / (np.dot(df['dU'], df['dU']))

        axs[1, 1].plot(df['U'].values / N, df['W'].values / N, '.', label=f"R = {R:.3}")
        axs[1, 1].plot(sorted(df['U'].values / N), sorted(df['dU'].values / N * Gamma + np.mean(df['W'].values / N)),
                       'r--', label=f"Gamma = {Gamma:.3}")
        axs[1, 1].set_xlabel('U/N')
        axs[1, 1].set_ylabel('W/N')
        axs[1, 1].legend()

    if __name__ == "__main__":
        plt.show(block=block)

    return



import numpy as np

# Implement Window: Rectangle/Hanning/Hamming
def build_windows(name='Hamming', N=20):
    # Default none
    window = None
    # Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    # Hanning
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    # Rectangle
    elif name == 'Rectangle':
        window = np.ones(N)
    return window



import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

# from pyphastft import fft


def main():
    fs = 100  # Sampling frequency (100 samples/second for this synthetic example)
    t_max = 6  # maximum time in "seconds"

    # Find the next lower power of 2 for the number of samples
    n_samples = 2 ** int(np.log2(t_max * fs))

    t = np.linspace(
        0, n_samples / fs, n_samples, endpoint=False
    )  # Adjusted time vector

    # Generate the signal
    s_re = 2 * np.sin(2 * np.pi * t + 1) + np.sin(2 * np.pi * 10 * t + 1)
    s_im = np.ascontiguousarray([0.0] * len(s_re), dtype=np.float64)

    # Plot the original signal
    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(t, s_re, label="f(x) = 2sin(x) + sin(10x)")
    plt.title("signal: f(x) = 2sin(x) + sin(10x)")
    plt.xlabel("time [seconds]")
    plt.ylabel("f(x)")
    plt.legend()

    # Perform FFT
    s_re = fft(s_re)

    # Plot the magnitude spectrum of the FFT result
    plt.subplot(2, 1, 2)
    plt.plot(
        np.abs(s_re),
        label="frequency spectrum",
    )
    plt.title("Signal after FFT")
    plt.xlabel("frequency (in Hz)")
    plt.ylabel("|FFT(f(x))|")

    # only show up to 11 Hz as in the wiki example
    plt.xlim(0, 11)

    plt.legend()
    plt.tight_layout()
    plt.savefig("wiki_fft_example.png", dpi=600)


if __name__ == "__main__":
    main()

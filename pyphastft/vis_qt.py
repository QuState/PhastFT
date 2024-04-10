import sys

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyphastft import fft
from pyqtgraph.Qt import QtWidgets, QtCore


class RealTimeAudioSpectrum(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(RealTimeAudioSpectrum, self).__init__(parent)
        self.n_fft_bins = 1024  # Increased FFT size for better frequency resolution
        self.n_display_bins = 32  # Maintain the same number of bars in the display
        self.sample_rate = 44100
        self.smoothing_factor = 0.1  # Smaller value for more smoothing
        self.ema_fft_data = np.zeros(
            self.n_display_bins
        )  # Adjusted to the number of display bins
        self.init_ui()
        self.init_audio_stream()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Customize plot aesthetics
        self.plot_widget.setBackground("k")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setTitle(
            "Real-Time Audio Spectrum Visualizer powered by PhastFT",
            color="w",
            size="16pt",
        )

        # Hide axis labels
        self.plot_item.getAxis("left").hide()
        self.plot_item.getAxis("bottom").hide()

        # Set fixed ranges for the x and y axes to prevent them from jumping
        self.plot_item.setXRange(0, self.sample_rate / 2, padding=0)
        self.plot_item.setYRange(0, 1, padding=0)

        self.bar_width = (
            (self.sample_rate / 2) / self.n_display_bins * 0.90
        )  # Adjusted width for display bins

        # Calculate bar positions so they are centered with respect to their frequency values
        self.freqs = np.linspace(
            0 + self.bar_width / 2,
            self.sample_rate / 2 - self.bar_width / 2,
            self.n_display_bins,
        )

        self.bar_graph = pg.BarGraphItem(
            x=self.freqs,
            height=np.zeros(self.n_display_bins),
            width=self.bar_width,
            brush=pg.mkBrush("m"),
        )
        self.plot_item.addItem(self.bar_graph)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update interval in milliseconds

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.n_fft_bins,  # This should match the FFT size
            stream_callback=self.audio_callback,
        )
        self.stream.start_stream()

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        reals = np.zeros(self.n_fft_bins)
        imags = np.zeros(self.n_fft_bins)
        reals[: len(audio_data)] = audio_data  # Fill the reals array with audio data
        fft(reals, imags, direction="f")
        fft_magnitude = np.sqrt(reals**2 + imags**2)[: self.n_fft_bins // 2]

        # Aggregate or interpolate FFT data to fit into display bins
        new_fft_data = np.interp(
            np.linspace(0, len(fft_magnitude), self.n_display_bins),
            np.arange(len(fft_magnitude)),
            fft_magnitude,
        )

        # Apply exponential moving average filter
        self.ema_fft_data = self.ema_fft_data * self.smoothing_factor + new_fft_data * (
            1.0 - self.smoothing_factor
        )
        return in_data, pyaudio.paContinue

    def update(self):
        self.bar_graph.setOpts(height=self.ema_fft_data, width=self.bar_width)

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeAudioSpectrum()
    window.show()
    sys.exit(app.exec_())

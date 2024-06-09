import sys
import numpy as np
import pyaudio
import pyqtgraph as pg
from pyphastft import rfft
from pyqtgraph.Qt import QtWidgets, QtCore

class RealTimeAudioSpectrum(QtWidgets.QWidget):
	def __init__(self, parent=None):
		super(RealTimeAudioSpectrum, self).__init__(parent)
		self.n_fft_bins = 1024
		self.n_display_bins = 64
		self.sample_rate = 44100
		self.smoothing_factor = 0.1  # Fine-tuned smoothing factor
		self.ema_fft_data = np.zeros(self.n_display_bins)
		self.init_ui()
		self.init_audio_stream()

	def init_ui(self):
		self.layout = QtWidgets.QVBoxLayout(self)
		self.plot_widget = pg.PlotWidget()
		self.layout.addWidget(self.plot_widget)

		self.plot_widget.setBackground("k")
		self.plot_item = self.plot_widget.getPlotItem()
		self.plot_item.setTitle(
				"Real-Time Audio Spectrum Visualizer powered by PhastFT",
				color="w",
				size="16pt",
				)

		self.plot_item.getAxis("left").hide()
		self.plot_item.getAxis("bottom").hide()

		self.plot_item.setXRange(0, self.sample_rate / 2, padding=0)
		self.plot_item.setYRange(0, 1, padding=0)

		self.bar_width = (self.sample_rate / 2) / self.n_display_bins * 0.8
		self.freqs = np.linspace(
				0, self.sample_rate / 2, self.n_display_bins, endpoint=False
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
		self.timer.start(50)

	def init_audio_stream(self):
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(
				format=pyaudio.paFloat32,
				channels=1,
				rate=self.sample_rate,
				input=True,
				frames_per_buffer=self.n_fft_bins,
				stream_callback=self.audio_callback,
				)
		self.stream.start_stream()

	def audio_callback(self, in_data, frame_count, time_info, status):
		audio_data = np.frombuffer(in_data, dtype=np.float32)
		audio_data = np.ascontiguousarray(audio_data, dtype=np.float64)
		reals, imags = rfft(audio_data, direction="f")
		reals = np.ascontiguousarray(reals)
		imags = np.ascontiguousarray(imags)
		fft_magnitude = np.sqrt(reals**2 + imags**2)[: self.n_fft_bins // 2]

		new_fft_data = np.interp(
				np.linspace(0, len(fft_magnitude), self.n_display_bins),
				np.arange(len(fft_magnitude)),
				fft_magnitude,
				)

		new_fft_data = np.log1p(new_fft_data)  # Apply logarithmic scaling

		self.ema_fft_data = self.ema_fft_data * self.smoothing_factor + new_fft_data * (
				1.0 - self.smoothing_factor
				)

		return in_data, pyaudio.paContinue

	def update(self):
		# Normalize the FFT data to ensure it fits within the display range
		max_value = np.max(self.ema_fft_data)
		if max_value > 0:
			normalized_fft_data = self.ema_fft_data / max_value
		else:
			normalized_fft_data = self.ema_fft_data

		self.bar_graph.setOpts(height=normalized_fft_data, width=self.bar_width)

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



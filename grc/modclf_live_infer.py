#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: shakir
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import numpy
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import modclf_live_infer_epy_block_2 as epy_block_2  # embedded python block
import numpy as np
import sip
import threading



class modclf_live_infer(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "modclf_live_infer")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 20
        self.samp_rate = samp_rate = 1e6
        self.cfo_hz = cfo_hz = 0
        self.sps = sps = 4
        self.span = span = 11
        self.samp_rate_0 = samp_rate_0 = 1e6
        self.rolloff = rolloff = 0.35
        self.noise_volt = noise_volt = (0.5 * 10**(-snr_db/10.0))**0.5
        self.mod_sel = mod_sel = 0
        self.cfo_norm = cfo_norm = cfo_hz / samp_rate

        ##################################################
        # Blocks
        ##################################################

        # Create the options list
        self._mod_sel_options = [0, 1, 2, 3]
        # Create the labels list
        self._mod_sel_labels = ['BPSK', 'QPSK', '8PSK', '16QAM']
        # Create the combo box
        self._mod_sel_tool_bar = Qt.QToolBar(self)
        self._mod_sel_tool_bar.addWidget(Qt.QLabel("Modulation" + ": "))
        self._mod_sel_combo_box = Qt.QComboBox()
        self._mod_sel_tool_bar.addWidget(self._mod_sel_combo_box)
        for _label in self._mod_sel_labels: self._mod_sel_combo_box.addItem(_label)
        self._mod_sel_callback = lambda i: Qt.QMetaObject.invokeMethod(self._mod_sel_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._mod_sel_options.index(i)))
        self._mod_sel_callback(self.mod_sel)
        self._mod_sel_combo_box.currentIndexChanged.connect(
            lambda i: self.set_mod_sel(self._mod_sel_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._mod_sel_tool_bar)
        self._snr_db_range = qtgui.Range(-5, 25, 1, 20, 200)
        self._snr_db_win = qtgui.RangeWidget(self._snr_db_range, self.set_snr_db, "SNR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._snr_db_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'Modulation (0 = BPSK, 1 = QPSK, 2 = 8PSK, 3 = 16QAM)', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-0.5, 3.5)

        self.qtgui_time_sink_x_0.set_y_label('Class Index', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.interp_fir_filter_xxx_0_0_0_0 = filter.interp_fir_filter_ccc(sps, firdes.root_raised_cosine(1.0, samp_rate, samp_rate/sps, rolloff, span*sps)
        )
        self.interp_fir_filter_xxx_0_0_0_0.declare_sample_delay(0)
        self.interp_fir_filter_xxx_0_0_0 = filter.interp_fir_filter_ccc(sps, firdes.root_raised_cosine(1.0, samp_rate, samp_rate/sps, rolloff, span*sps)
        )
        self.interp_fir_filter_xxx_0_0_0.declare_sample_delay(0)
        self.interp_fir_filter_xxx_0_0 = filter.interp_fir_filter_ccc(sps, firdes.root_raised_cosine(1.0, samp_rate, samp_rate/sps, rolloff, span*sps)
        )
        self.interp_fir_filter_xxx_0_0.declare_sample_delay(0)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccc(sps, firdes.root_raised_cosine(1.0, samp_rate, samp_rate/sps, rolloff, span*sps)
        )
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.epy_block_2 = epy_block_2.blk(model_path="/media/shakir/EXT4_1TB/projects/sdr/sdr-ModRec/models/modclf.onnx", labels_path="/media/shakir/EXT4_1TB/projects/sdr/sdr-ModRec/models/labels.json", win_len=1024, hop_len=256, print_on_change=True, enable_label_out=True, enable_popup=True, popup_title="Predicted Modulation")
        self.digital_chunks_to_symbols_xx_0_0_0_0 = digital.chunks_to_symbols_bc([(i+1j*q)/np.sqrt(10) for q in [-3,-1,1,3] for i in [-3,-1,1,3]], 2)
        self.digital_chunks_to_symbols_xx_0_0_0 = digital.chunks_to_symbols_bc([np.exp(1j*2*np.pi*k/8) for k in range(8)], 2)
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc([(1+1j)/np.sqrt(2), (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2), (1-1j)/np.sqrt(2)], 2)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc([-1+0j, 1+0j], 2)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_volt,
            frequency_offset=cfo_norm,
            epsilon=1.0,
            taps=[1.0+0j],
            noise_seed=0,
            block_tags=False)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_selector_0 = blocks.selector(gr.sizeof_gr_complex*1,mod_sel,0)
        self.blocks_selector_0.set_enabled(True)
        self.analog_random_source_x_0_0_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 15, 1000))), True)
        self.analog_random_source_x_0_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 7, 1000))), True)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 3, 1000))), True)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 1, 1000))), True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.analog_random_source_x_0_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.analog_random_source_x_0_0_0, 0), (self.digital_chunks_to_symbols_xx_0_0_0, 0))
        self.connect((self.analog_random_source_x_0_0_0_0, 0), (self.digital_chunks_to_symbols_xx_0_0_0_0, 0))
        self.connect((self.blocks_selector_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.epy_block_2, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.interp_fir_filter_xxx_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0, 0), (self.interp_fir_filter_xxx_0_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0_0, 0), (self.interp_fir_filter_xxx_0_0_0_0, 0))
        self.connect((self.epy_block_2, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.blocks_selector_0, 0))
        self.connect((self.interp_fir_filter_xxx_0_0, 0), (self.blocks_selector_0, 1))
        self.connect((self.interp_fir_filter_xxx_0_0_0, 0), (self.blocks_selector_0, 2))
        self.connect((self.interp_fir_filter_xxx_0_0_0_0, 0), (self.blocks_selector_0, 3))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "modclf_live_infer")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.set_noise_volt((0.5 * 10**(-self.snr_db/10.0))**0.5)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_cfo_norm(self.cfo_hz / self.samp_rate)
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.interp_fir_filter_xxx_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_cfo_hz(self):
        return self.cfo_hz

    def set_cfo_hz(self, cfo_hz):
        self.cfo_hz = cfo_hz
        self.set_cfo_norm(self.cfo_hz / self.samp_rate)

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.interp_fir_filter_xxx_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )

    def get_span(self):
        return self.span

    def set_span(self, span):
        self.span = span
        self.interp_fir_filter_xxx_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )

    def get_samp_rate_0(self):
        return self.samp_rate_0

    def set_samp_rate_0(self, samp_rate_0):
        self.samp_rate_0 = samp_rate_0

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff
        self.interp_fir_filter_xxx_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )
        self.interp_fir_filter_xxx_0_0_0_0.set_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.samp_rate/self.sps, self.rolloff, self.span*self.sps)
        )

    def get_noise_volt(self):
        return self.noise_volt

    def set_noise_volt(self, noise_volt):
        self.noise_volt = noise_volt
        self.channels_channel_model_0.set_noise_voltage(self.noise_volt)

    def get_mod_sel(self):
        return self.mod_sel

    def set_mod_sel(self, mod_sel):
        self.mod_sel = mod_sel
        self._mod_sel_callback(self.mod_sel)
        self.blocks_selector_0.set_input_index(self.mod_sel)

    def get_cfo_norm(self):
        return self.cfo_norm

    def set_cfo_norm(self, cfo_norm):
        self.cfo_norm = cfo_norm
        self.channels_channel_model_0.set_frequency_offset(self.cfo_norm)




def main(top_block_cls=modclf_live_infer, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()

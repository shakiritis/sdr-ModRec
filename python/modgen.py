#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: modclf_dataset_gen.grc
# Author: shakir
# GNU Radio version: 3.10.12.0

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
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import numpy as np
import threading




class modgen(gr.top_block):

    def __init__(self, cfo_hz=0, mod_sel=0, n_samps=300000, out_file="/media/shakir/EXT4_1TB/projects/sdr/modclf-gnuradio-ml/data/raw/mod0_snr10_cfo0.c64", snr_db=10):
        gr.top_block.__init__(self, "modclf_dataset_gen.grc", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.cfo_hz = cfo_hz
        self.mod_sel = mod_sel
        self.n_samps = n_samps
        self.out_file = out_file
        self.snr_db = snr_db

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e6
        self.sps = sps = 4
        self.span = span = 11
        self.rolloff = rolloff = 0.35
        self.noise_volt = noise_volt = (0.5 * 10**(-snr_db/10.0))**0.5
        self.cfo_norm = cfo_norm = cfo_hz / samp_rate

        ##################################################
        # Blocks
        ##################################################

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
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, n_samps)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, out_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)
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
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_selector_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.blocks_head_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.interp_fir_filter_xxx_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0, 0), (self.interp_fir_filter_xxx_0_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0_0, 0), (self.interp_fir_filter_xxx_0_0_0_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.blocks_selector_0, 0))
        self.connect((self.interp_fir_filter_xxx_0_0, 0), (self.blocks_selector_0, 1))
        self.connect((self.interp_fir_filter_xxx_0_0_0, 0), (self.blocks_selector_0, 2))
        self.connect((self.interp_fir_filter_xxx_0_0_0_0, 0), (self.blocks_selector_0, 3))


    def get_cfo_hz(self):
        return self.cfo_hz

    def set_cfo_hz(self, cfo_hz):
        self.cfo_hz = cfo_hz
        self.set_cfo_norm(self.cfo_hz / self.samp_rate)

    def get_mod_sel(self):
        return self.mod_sel

    def set_mod_sel(self, mod_sel):
        self.mod_sel = mod_sel
        self.blocks_selector_0.set_input_index(self.mod_sel)

    def get_n_samps(self):
        return self.n_samps

    def set_n_samps(self, n_samps):
        self.n_samps = n_samps
        self.blocks_head_0.set_length(self.n_samps)

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.blocks_file_sink_0.open(self.out_file)

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

    def get_cfo_norm(self):
        return self.cfo_norm

    def set_cfo_norm(self, cfo_norm):
        self.cfo_norm = cfo_norm
        self.channels_channel_model_0.set_frequency_offset(self.cfo_norm)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfo-hz", dest="cfo_hz", type=eng_float, default=eng_notation.num_to_str(float(0)),
        help="Set cfo_hz [default=%(default)r]")
    parser.add_argument(
        "--mod-sel", dest="mod_sel", type=intx, default=0,
        help="Set mod_sel [default=%(default)r]")
    parser.add_argument(
        "--n-samps", dest="n_samps", type=intx, default=300000,
        help="Set n_samps [default=%(default)r]")
    parser.add_argument(
        "--out-file", dest="out_file", type=str, default="/media/shakir/EXT4_1TB/projects/sdr/modclf-gnuradio-ml/data/raw/mod0_snr10_cfo0.c64",
        help="Set out_file [default=%(default)r]")
    parser.add_argument(
        "--snr-db", dest="snr_db", type=eng_float, default=eng_notation.num_to_str(float(10)),
        help="Set snr_db [default=%(default)r]")
    return parser


def main(top_block_cls=modgen, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(cfo_hz=options.cfo_hz, mod_sel=options.mod_sel, n_samps=options.n_samps, out_file=options.out_file, snr_db=options.snr_db)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()

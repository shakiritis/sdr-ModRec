import numpy as np
import onnxruntime as ort
import json
import pmt
from gnuradio import gr

from PyQt5 import QtWidgets, QtCore

class LabelPopup(QtWidgets.QWidget):
    def __init__(self, title="Modulation Prediction"):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(420, 160)

        self.label = QtWidgets.QLabel("...", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        font = self.label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.label.setFont(font)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    @QtCore.pyqtSlot(str)
    def set_text(self, text):
        self.label.setText(text)

class blk(gr.sync_block):
    def __init__(self,
                 model_path="models/modclf.onnx",
                 labels_path="models/labels.json",
                 win_len=1024,
                 hop_len=256,
                 print_on_change=True,
                 enable_label_out=True,
                 enable_popup=True,
                 popup_title="Modulation Prediction"):
        gr.sync_block.__init__(
            self,
            name="modclf_infer_onnx",
            in_sig=[np.complex64],
            out_sig=[np.float32],   # ONLY label index output
        )

        # Optional message port
        self.enable_label_out = bool(enable_label_out)
        if self.enable_label_out:
            self.message_port_register_out(pmt.intern("label_out"))

        self.win_len = int(win_len)
        self.hop_len = int(hop_len)
        self.print_on_change = bool(print_on_change)

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

        self.buf = np.zeros(self.win_len, dtype=np.complex64)
        self.fill = 0
        self.last_pred = -1
        self.current_pred = 0.0

        # Popup
        self._popup = None
        self.enable_popup = bool(enable_popup)
        if self.enable_popup:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                self._popup = LabelPopup(title=str(popup_title))
                self._popup.show()

    def _preprocess(self, x_cplx):
        mag2 = (np.real(x_cplx).astype(np.float64)**2 + np.imag(x_cplx).astype(np.float64)**2)
        p = np.mean(mag2)
        if (not np.isfinite(p)) or p < 1e-18:
            return None
        x = x_cplx / np.float32(np.sqrt(p))

        I = np.real(x).astype(np.float32)
        Q = np.imag(x).astype(np.float32)
        X = np.stack([I, Q], axis=0)[None, :, :]  # (1,2,L)
        return X

    def _infer(self, x_cplx):
        X = self._preprocess(x_cplx)
        if X is None:
            return None
        logits = self.sess.run(None, {self.input_name: X})[0]  # (1,C)
        return int(np.argmax(logits, axis=1)[0])

    def _popup_set_label(self, label):
        if self._popup is not None:
            QtCore.QMetaObject.invokeMethod(
                self._popup,
                "set_text",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(label))
            )

    def work(self, input_items, output_items):
        x = input_items[0]
        ylab = output_items[0]  # float label index output

        # hold last prediction across samples
        ylab[:] = self.current_pred

        n = len(x)
        i = 0
        while i < n:
            take = min(n - i, self.win_len - self.fill)
            self.buf[self.fill:self.fill+take] = x[i:i+take]
            self.fill += take
            i += take

            if self.fill == self.win_len:
                pred = self._infer(self.buf)
                if pred is not None:
                    self.current_pred = float(pred)

                    if pred != self.last_pred:
                        label = self.labels[pred]

                        if self.print_on_change:
                            print("[modclf] predicted:", label)

                        self._popup_set_label(label)

                        if self.enable_label_out:
                            self.message_port_pub(pmt.intern("label_out"), pmt.intern(label))

                        self.last_pred = pred

                # overlap shift
                hop = self.hop_len
                self.buf[:self.win_len-hop] = self.buf[hop:]
                self.fill = self.win_len - hop

        return n


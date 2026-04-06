"""
FourierLab — Educational Fourier Analysis & FFT Visualization
=============================================================
Course: Engineering Analysis 2025-2026
Developer: Bnar Haje

PySide6 — Dark Theme · Colored Graphs · Step-by-Step Animation
Every series plot animates from n = 1 → target n, one harmonic at a time.

Install:
    pip install PySide6 matplotlib numpy
"""

import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QFrame,
    QStackedWidget, QSizePolicy, QGroupBox, QGridLayout,
    QCheckBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QPalette

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ═══════════════════════════════════════════════════════
#  DARK PALETTE
# ═══════════════════════════════════════════════════════
BG       = "#0e0e1a"
BG_CARD  = "#141425"
BG_INPUT = "#1a1a30"
BG_BTN   = "#1e1e3a"
BORDER   = "#2a2a50"
ACCENT   = "#5c6bc0"
GREEN    = "#00e676"
TEAL     = "#26c6da"
ORANGE   = "#ff9100"
RED      = "#ff5252"
PURPLE   = "#b388ff"
PINK     = "#f06292"
TEXT     = "#e0e0f0"
TEXT2    = "#9098c0"
MUTED    = "#555580"
PLOT_BG  = "#0a0a14"
GRID     = "#1c1c34"

plt.rcParams.update({
    "figure.facecolor": PLOT_BG, "axes.facecolor": PLOT_BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "grid.color": GRID, "grid.alpha": 0.5, "font.size": 10,
})

# ═══════════════════════════════════════════════════════
#  WAVEFORM & SIGNAL DATA
# ═══════════════════════════════════════════════════════
WAVEFORMS = [
    {"name": "Square Wave",    "key": "square",   "T": 4.0},
    {"name": "Sawtooth Wave",  "key": "sawtooth", "T": 4.0},
    {"name": "Triangle Wave",  "key": "triangle", "T": 4.0},
    {"name": "Half-Rect Sine", "key": "halfrect", "T": 4.0},
    {"name": "Full-Rect Sine", "key": "fullrect", "T": 2.0},
    {"name": "Pulse Train",    "key": "pulse",    "T": 4.0},
    {"name": "Ramp Function",  "key": "ramp",     "T": 4.0},
]
FFT_SIGNALS = [
    {"name": "Pure Sine 50 Hz",       "key": "sine50",   "fs": 1000, "dur": 0.2},
    {"name": "Multi-Sine 50+120+200", "key": "multisin", "fs": 2000, "dur": 0.1},
    {"name": "AM Signal",             "key": "am",       "fs": 2000, "dur": 0.2},
    {"name": "FM Signal",             "key": "fm",       "fs": 2000, "dur": 0.2},
    {"name": "Square 50 Hz",          "key": "sq50",     "fs": 5000, "dur": 0.1},
    {"name": "Chirp 10-400 Hz",       "key": "chirp",    "fs": 2000, "dur": 0.5},
    {"name": "Noisy Sine 50 Hz",      "key": "noisysine","fs": 1000, "dur": 0.2},
]

# ═══════════════════════════════════════════════════════
#  MATH
# ═══════════════════════════════════════════════════════
def get_wave(key, x, T):
    L = T / 2; xn = ((x + L) % T) - L
    if   key == "square":   return np.where(xn >= 0, 1.0, -1.0)
    elif key == "sawtooth": return xn / L
    elif key == "triangle": return 1.0 - 2.0 * np.abs(xn) / L
    elif key == "halfrect": return np.maximum(np.sin(np.pi * xn / L), 0.0)
    elif key == "fullrect": return np.abs(np.sin(np.pi * xn / L))
    elif key == "pulse":    return np.where(np.abs(xn) <= L * 0.28, 1.0, 0.0)
    elif key == "ramp":     return (xn + L) / (2 * L)
    return np.zeros_like(x)

def compute_coeffs(key, T, N_max=80):
    L = T / 2; Np = 8192
    x = np.linspace(-L, L, Np, endpoint=False); dx = x[1] - x[0]
    fx = get_wave(key, x, T)
    a0 = np.sum(fx) * dx / (2 * L)
    ns = np.arange(1, N_max + 1)
    ph = ns[None, :] * np.pi * x[:, None] / L
    fX = fx[:, None]
    an = np.sum(fX * np.cos(ph), axis=0) * dx / L
    bn = np.sum(fX * np.sin(ph), axis=0) * dx / L
    return a0, an, bn

def reconstruct(x, a0, an, bn, L, n):
    if n == 0: return np.full(len(x), float(a0))
    ns = np.arange(1, n + 1)
    ph = ns[None, :] * np.pi * x[:, None] / L
    return float(a0) + (an[:n][None, :] * np.cos(ph) + bn[:n][None, :] * np.sin(ph)).sum(1)

def get_fft_signal(key, fs, dur):
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    if   key == "sine50":   y = np.sin(2*np.pi*50*t)
    elif key == "multisin": y = np.sin(2*np.pi*50*t)+.6*np.sin(2*np.pi*120*t)+.3*np.sin(2*np.pi*200*t)
    elif key == "am":       y = (1+.8*np.sin(2*np.pi*20*t))*np.sin(2*np.pi*200*t)
    elif key == "fm":       y = np.sin(2*np.pi*(50*t+15*np.sin(2*np.pi*5*t)))
    elif key == "sq50":     y = np.sign(np.sin(2*np.pi*50*t)).astype(float)
    elif key == "chirp":
        fi = 10+(400-10)*t/dur; y = np.sin(2*np.pi*np.cumsum(fi)/fs)
    elif key == "noisysine":
        rng = np.random.default_rng(42); y = np.sin(2*np.pi*50*t)+.5*rng.standard_normal(len(t))
    else: y = np.zeros_like(t)
    return t, y

# ═══════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════
SS = f"""
QMainWindow, QWidget#central {{ background: {BG}; }}
QWidget {{
    font-family: "SF Pro Display","Helvetica Neue","Segoe UI","Arial",sans-serif;
    font-size: 12px; color: {TEXT};
}}
QPushButton {{
    background: {BG_BTN}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 6px 16px; font-size: 12px; min-height: 24px;
}}
QPushButton:hover {{ background: #2a2a52; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; }}
QPushButton#plot_btn {{
    background: #1a2848; border: 1px solid #3355aa; color: #88bbff; font-weight: bold;
}}
QPushButton#plot_btn:hover {{ background: #253660; border-color: #5588dd; }}
QPushButton#action_btn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1b3a1b, stop:1 #1b2a3a);
    border: 1px solid #338833; color: {GREEN}; font-weight: bold;
}}
QPushButton#action_btn:hover {{ background: #224422; border-color: {GREEN}; }}
QPushButton#spectrum_btn {{
    background: {BG_BTN}; border: 1px solid {BORDER}; color: {TEAL};
}}
QPushButton#spectrum_btn:hover {{ border-color: {TEAL}; background: #1a2a3a; }}
QLineEdit {{
    background: {BG_INPUT}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 3px; padding: 4px 6px; selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox {{
    background: {BG_INPUT}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 3px; padding: 5px 10px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {BG_CARD}; color: {TEXT};
    selection-background-color: {ACCENT}; border: 1px solid {BORDER};
}}
QDoubleSpinBox {{
    background: {BG_INPUT}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 3px; padding: 3px 6px;
}}
QCheckBox {{ color: {TEXT2}; spacing: 5px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {BORDER};
    border-radius: 3px; background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QGroupBox {{
    font-weight: bold; font-size: 11px; color: {TEAL};
    border: 1px solid {BORDER}; border-radius: 6px;
    margin-top: 10px; padding-top: 16px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QLabel {{ color: {TEXT2}; background: transparent; }}
QLabel#heading {{ color: {TEXT}; font-size: 11px; font-weight: bold; }}
QFrame#plot_frame {{
    background: {PLOT_BG}; border: 1px solid {BORDER}; border-radius: 6px;
}}
QFrame#formula_frame, QFrame#series_frame {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 6px;
}}
QFrame#sidebar {{ background: {BG_CARD}; border-right: 1px solid {BORDER}; }}
QPushButton#nav_btn {{
    background: transparent; border: none; border-radius: 5px;
    padding: 8px 14px; text-align: left; font-size: 12px; color: {TEXT2};
}}
QPushButton#nav_btn:hover {{ background: #1e1e40; }}
QPushButton#nav_btn:checked {{
    background: #252550; color: #ffffff; font-weight: bold;
    border-left: 3px solid {GREEN};
}}
"""

# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def init_ax(ax):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(True, alpha=0.4, color=GRID, lw=0.5)
    for s in ax.spines.values(): s.set_color(GRID); s.set_linewidth(0.5)

def _e(v="", w=50):
    e = QLineEdit(v); e.setFixedWidth(w); return e

def _l(t): return QLabel(t)

def _pb(t):
    b = QPushButton(t); b.setObjectName("plot_btn"); b.setCursor(Qt.PointingHandCursor); return b

def _ab(t):
    b = QPushButton(t); b.setObjectName("action_btn"); b.setCursor(Qt.PointingHandCursor); return b

def _sb(t):
    b = QPushButton(t); b.setObjectName("spectrum_btn"); b.setCursor(Qt.PointingHandCursor); return b

def _nb(t):
    b = QPushButton(t); b.setObjectName("nav_btn"); b.setCheckable(True)
    b.setCursor(Qt.PointingHandCursor); b.setFixedHeight(36); return b

def _cv(fig, lay):
    c = FigureCanvas(fig); c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lay.addWidget(c); return c

# ═══════════════════════════════════════════════════════
#  FORMULA WIDGET
# ═══════════════════════════════════════════════════════
class FormulaWidget(FigureCanvas):
    def __init__(self, w=4.0, h=2.4):
        self._fig = Figure(figsize=(w, h), dpi=100, facecolor=BG_CARD)
        super().__init__(self._fig)
        self.ax = self._fig.add_axes([0, 0, 1, 1])
        self._reset()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _reset(self):
        self.ax.clear(); self.ax.axis("off")
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.set_facecolor(BG_CARD)

    def show_definitions(self):
        self._reset()
        kw = dict(fontsize=13, color=TEXT, va="center")
        self.ax.text(0.04, 0.86,
            r"$f(x)=a_0+\sum_{n=1}^{\infty}"
            r"\!\left(a_n\cos\frac{n\pi}{L}x+b_n\sin\frac{n\pi}{L}x\right)$", **kw)
        self.ax.text(0.04, 0.64,
            r"$a_0=\frac{1}{2L}\int_{-L}^{L}f(x)\,dx$", **kw)
        self.ax.text(0.04, 0.42,
            r"$a_n=\frac{1}{L}\int_{-L}^{L}f(x)\cos\frac{n\pi}{L}x\,dx$", **kw)
        self.ax.text(0.04, 0.20,
            r"$b_n=\frac{1}{L}\int_{-L}^{L}f(x)\sin\frac{n\pi}{L}x\,dx$", **kw)
        self.draw()

    def show_series(self, a0, an, bn, L, nshow=4):
        self._reset()
        terms = []
        if abs(a0) > 1e-5:
            terms.append(f"{a0:.4f}")
        for i in range(1, min(nshow + 1, len(an) + 1)):
            a, b = an[i - 1], bn[i - 1]
            if abs(a) > 1e-5:
                s = "+" if a > 0 and terms else ""
                terms.append(fr"{s}\frac{{{abs(a):.3f}\cos\frac{{{i}\pi x}}{{{L:.3g}}}}}{{1}}")
            if abs(b) > 1e-5:
                s = "+" if b > 0 and terms else ""
                terms.append(fr"{s}\frac{{{abs(b):.3f}\sin\frac{{{i}\pi x}}{{{L:.3g}}}}}{{1}}")
        tex = r"$" + r"\;".join(terms[:8]) + r"\;\cdots$" if terms else r"$f(x)=0$"
        try:
            self.ax.text(0.50, 0.50, tex, fontsize=13, color=TEXT, va="center", ha="center")
        except Exception:
            self.ax.text(0.50, 0.50, " ".join(terms[:6]),
                         fontsize=10, color=TEXT, va="center", ha="center", family="monospace")
        self.draw()


# ═══════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════
class FourierLab(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Expansion by Bnar Haje")
        self.resize(1340, 860); self.setMinimumSize(QSize(1060, 680))
        self.setStyleSheet(SS)

        cw = QWidget(); cw.setObjectName("central")
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self._build_sidebar(root)
        self.stack = QStackedWidget(); root.addWidget(self.stack, 1)

        # coefficient cache
        self._cache_key = self._cache_T = None
        self._a0 = self._an = self._bn = None

        # ═══════════════════════════════════════
        #  ANIMATION STATE
        #  _anim_cur:    harmonic currently shown
        #  _anim_target: harmonic we're animating toward
        #  _anim_paused: True when user paused mid-animation
        # ═══════════════════════════════════════
        self._anim_cur    = 0
        self._anim_target = 0
        self._anim_paused = False
        self._anim_timer  = QTimer()
        self._anim_timer.timeout.connect(self._anim_step)

        self._build_series_page()
        self._build_spectrum_page()
        self._build_fft_page()

        self._set_nav("series"); self.stack.setCurrentIndex(0)
        # animate on startup
        QTimer.singleShot(150, self._start_animation)

    # ─────────── sidebar ───────────
    def _build_sidebar(self, root):
        sb = QFrame(); sb.setObjectName("sidebar"); sb.setFixedWidth(200)
        sl = QVBoxLayout(sb); sl.setContentsMargins(10, 16, 10, 12); sl.setSpacing(2)
        t = QLabel("∿  FourierLab")
        t.setStyleSheet(f"font-size:17px;font-weight:bold;color:{ACCENT};padding:6px 2px;")
        sl.addWidget(t)
        sl.addWidget(QLabel("Engineering Analysis 2025-26"))
        sl.addSpacing(12)
        self.nav = {}
        for key, txt, i in [("series","∿  Fourier Series",0),
                            ("spectrum","📊  Spectrum",1),
                            ("fft","⚡  FFT Analysis",2)]:
            b = _nb(txt)
            b.clicked.connect(lambda _, k=key, idx=i: self._go(k, idx))
            sl.addWidget(b); self.nav[key] = b
        sl.addStretch()
        by = QLabel("by Bnar Haje")
        by.setStyleSheet(f"color:{TEXT2};font-size:11px;padding:2px;")
        sl.addWidget(by)
        root.addWidget(sb)

    def _go(self, k, i):
        self._stop_animation(); self._set_nav(k); self.stack.setCurrentIndex(i)

    def _set_nav(self, k):
        for n, b in self.nav.items(): b.setChecked(n == k)

    # ═══════════════════════════════════════════
    #  PAGE 1 — FOURIER SERIES
    # ═══════════════════════════════════════════
    def _build_series_page(self):
        page = QWidget()
        vl = QVBoxLayout(page); vl.setContentsMargins(10, 8, 10, 6); vl.setSpacing(6)

        # ── PLOT ──
        pf = QFrame(); pf.setObjectName("plot_frame")
        pfl = QVBoxLayout(pf); pfl.setContentsMargins(2, 2, 2, 2)
        self.s_fig = Figure(figsize=(9, 4), dpi=100, facecolor=PLOT_BG)
        self.s_ax = self.s_fig.add_subplot(111); init_ax(self.s_ax)
        self.s_canvas = _cv(self.s_fig, pfl)
        vl.addWidget(pf, 5)

        # ── CONTROLS + FORMULAS ──
        bot = QHBoxLayout(); bot.setSpacing(10)
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(5)

        # Row 1:  [Plot 1]  n  T  x Periods
        r1 = QHBoxLayout(); r1.setSpacing(6)
        b1 = _pb("Plot 1"); b1.clicked.connect(self._start_animation); r1.addWidget(b1)
        r1.addWidget(_l("n"));  self.s_n = _e("3", 42);  r1.addWidget(self.s_n)
        r1.addWidget(_l("T"));  self.s_T = _e("4", 42);  r1.addWidget(self.s_T)
        r1.addWidget(_l("x Periods")); self.s_xp = _e("-2 -1 1 2", 100); r1.addWidget(self.s_xp)
        r1.addStretch(); cl.addLayout(r1)

        # Row 2:  [Plot 2]  Xmin  Xmax  y function  Plot f
        r2 = QHBoxLayout(); r2.setSpacing(6)
        b2 = _pb("Plot 2"); b2.clicked.connect(self._start_animation); r2.addWidget(b2)
        r2.addWidget(_l("Xmin")); self.s_xmin = _e("-2", 42); r2.addWidget(self.s_xmin)
        r2.addWidget(_l("Xmax")); self.s_xmax = _e("11", 42); r2.addWidget(self.s_xmax)
        r2.addWidget(_l("y function")); self.s_yfunc = _e("0 1 0", 100); r2.addWidget(self.s_yfunc)
        self.s_plotf = QCheckBox("Plot f"); r2.addWidget(self.s_plotf)
        r2.addStretch(); cl.addLayout(r2)

        # Row 3:  [Plot 3]  Ymin  Ymax  Speed  Grid
        r3 = QHBoxLayout(); r3.setSpacing(6)
        b3 = _pb("Plot 3"); b3.clicked.connect(self._start_animation); r3.addWidget(b3)
        r3.addWidget(_l("Ymin")); self.s_ymin = _e("-1.5", 42); r3.addWidget(self.s_ymin)
        r3.addWidget(_l("Ymax")); self.s_ymax = _e("1.5", 42); r3.addWidget(self.s_ymax)
        r3.addWidget(_l("Speed"))
        self.s_speed = QDoubleSpinBox()
        self.s_speed.setRange(0.01, 5.0); self.s_speed.setValue(0.1)
        self.s_speed.setSingleStep(0.05); self.s_speed.setFixedWidth(68)
        r3.addWidget(self.s_speed)
        self.s_grid = QCheckBox("Grid"); self.s_grid.setChecked(True)
        r3.addWidget(self.s_grid)
        r3.addStretch(); cl.addLayout(r3)

        # Row 4:  [Pause]  [Save]  [Compute]  x  Result
        r4 = QHBoxLayout(); r4.setSpacing(6)
        self.s_pause = QPushButton("Pause")
        self.s_pause.clicked.connect(self._toggle_pause); r4.addWidget(self.s_pause)
        self.s_save = QPushButton("Save")
        self.s_save.clicked.connect(self._save_figure); r4.addWidget(self.s_save)
        self.s_comp = QPushButton("Compute")
        self.s_comp.clicked.connect(self._eval_x0); r4.addWidget(self.s_comp)
        r4.addWidget(_l("x")); self.s_x0 = _e("1", 42); r4.addWidget(self.s_x0)
        r4.addWidget(_l("Result"))
        self.s_result = QLineEdit(); self.s_result.setReadOnly(True); self.s_result.setFixedWidth(170)
        r4.addWidget(self.s_result)
        r4.addStretch(); cl.addLayout(r4)

        # auto-update result whenever n or x changes
        self.s_n.textChanged.connect(self._eval_x0)
        self.s_x0.textChanged.connect(self._eval_x0)
        # pressing Enter in the n-field restarts the animation with the new n
        self.s_n.returnPressed.connect(self._start_animation)
        # editing n then clicking away redraws at that n immediately
        self.s_n.editingFinished.connect(self._draw_at_current_n)

        # Row 5:  Spectrum buttons + Waveform combo
        r5 = QHBoxLayout(); r5.setSpacing(6)
        sa = _sb("Spectrum an");  sa.clicked.connect(lambda: self._jump_sp("an"));  r5.addWidget(sa)
        sb_ = _sb("Spectrum bn"); sb_.clicked.connect(lambda: self._jump_sp("bn")); r5.addWidget(sb_)
        sc = _sb("Spectrum amplitude"); sc.clicked.connect(lambda: self._jump_sp("amplitude")); r5.addWidget(sc)
        r5.addStretch()
        r5.addWidget(_l("Waveform"))
        self.s_combo = QComboBox()
        self.s_combo.addItems([w["name"] for w in WAVEFORMS])
        self.s_combo.setFixedWidth(160)
        self.s_combo.currentIndexChanged.connect(self._on_wave)
        r5.addWidget(self.s_combo)
        cl.addLayout(r5)

        bot.addWidget(ctrl, 5)

        # Formula panel (right)
        ff = QFrame(); ff.setObjectName("formula_frame")
        ffl = QVBoxLayout(ff); ffl.setContentsMargins(4, 4, 4, 4)
        self.formula_w = FormulaWidget(4.4, 2.6)
        self.formula_w.show_definitions()
        ffl.addWidget(self.formula_w)
        bot.addWidget(ff, 3)
        vl.addLayout(bot, 3)

        # Series result bar (bottom)
        sf = QFrame(); sf.setObjectName("series_frame")
        sfl = QHBoxLayout(sf); sfl.setContentsMargins(4, 2, 4, 2)
        self.series_w = FormulaWidget(9, 0.9)
        sfl.addWidget(self.series_w)
        vl.addWidget(sf, 1)

        self.stack.addWidget(page)

    # ─── helpers ───
    def _on_wave(self, i):
        self.s_T.setText(str(WAVEFORMS[i]["T"]))
        self._start_animation()

    def _fval(self, le, default):
        try: return float(le.text())
        except: return default

    def _target_n(self):
        try: return max(1, int(self.s_n.text()))
        except: return 5

    def _ensure_coeffs(self):
        key = WAVEFORMS[self.s_combo.currentIndex()]["key"]
        T = self._fval(self.s_T, 4.0)
        if key != self._cache_key or T != self._cache_T:
            self._a0, self._an, self._bn = compute_coeffs(key, T, 80)
            self._cache_key, self._cache_T = key, T
        return key, T

    # ═══════════════════════════════════════════
    #  ANIMATION ENGINE
    #
    #  _start_animation()   → resets to n=0, starts timer
    #  _anim_step()         → called each tick: n++ then draw
    #  _toggle_pause()      → pause / resume / restart
    #  _stop_animation()    → hard stop
    #
    #  Every Plot button, every waveform change calls
    #  _start_animation() so the user always sees n=1,
    #  n=2, n=3 … building up step by step.
    # ═══════════════════════════════════════════

    def _start_animation(self):
        """Animate from n=1 up to the target n, one step at a time."""
        self._anim_timer.stop()
        self._ensure_coeffs()
        self._anim_cur    = 0               # next tick will show n=1
        self._anim_target = self._target_n()
        self._anim_paused = False
        self.s_pause.setText("Pause")
        # interval from Speed spinbox (seconds → ms, minimum 40ms)
        ms = max(40, int(1000 * self.s_speed.value()))
        self._anim_timer.start(ms)

    def _anim_step(self):
        """One timer tick: advance n by 1, draw the frame."""
        self._anim_cur += 1
        n = self._anim_cur

        # reached target → stop cleanly
        if n > self._anim_target:
            self._anim_timer.stop()
            self.s_pause.setText("Play")
            return

        key = WAVEFORMS[self.s_combo.currentIndex()]["key"]
        T   = self._fval(self.s_T, 4.0)
        L   = T / 2

        # show current step in the n field
        self.s_n.setText(str(n))

        # draw this harmonic step
        self._draw_frame(key, T, n)

        # update series formula at bottom
        self.series_w.show_series(self._a0, self._an, self._bn, L, min(n, 5))

    def _toggle_pause(self):
        if self._anim_timer.isActive():
            # playing → pause
            self._anim_timer.stop()
            self._anim_paused = True
            self.s_pause.setText("Play")
        elif self._anim_paused and self._anim_cur < self._anim_target:
            # paused mid-way → resume from where we stopped
            ms = max(40, int(1000 * self.s_speed.value()))
            self._anim_timer.start(ms)
            self._anim_paused = False
            self.s_pause.setText("Pause")
        else:
            # finished or never started → restart from n=1
            self._start_animation()

    def _stop_animation(self):
        self._anim_timer.stop()
        self._anim_paused = False

    # ─── draw one frame ───
    def _draw_frame(self, key, T, n):
        L    = T / 2
        xmin = self._fval(self.s_xmin, -2)
        xmax = self._fval(self.s_xmax, 11)
        ymin = self._fval(self.s_ymin, -1.5)
        ymax = self._fval(self.s_ymax, 1.5)

        x  = np.linspace(xmin, xmax, 3000)
        yo = get_wave(key, x, T)
        yr = reconstruct(x, self._a0, self._an, self._bn, L, n)

        ax = self.s_ax; ax.clear(); init_ax(ax)

        if self.s_grid.isChecked():
            ax.grid(True, alpha=0.35, color=GRID, lw=0.5)
        else:
            ax.grid(False)

        # original waveform (dim gray outline)
        ax.plot(x, yo, color="#555580", lw=1.2, alpha=0.65)
        if self.s_plotf.isChecked():
            ax.plot(x, yo, color=TEXT2, lw=1.6, label="f(x)")

        # Fourier reconstruction — bright colored
        ax.plot(x, yr, color="#5cb8ff", lw=2.3, label=f"Fourier  n = {n}")

        # dashed red reference lines
        ax.axhline(0, color=RED, lw=0.7, ls="--", alpha=0.4)
        ax.axvline(0, color=RED, lw=0.7, ls="--", alpha=0.4)

        # n = label (large, top-left)
        ax.annotate(f"n = {n}", xy=(0.02, 0.92), xycoords="axes fraction",
                    fontsize=16, fontweight="bold", color=TEXT)

        # RMS error
        rms = np.sqrt(np.mean((yo - yr) ** 2))
        ax.annotate(f"RMS = {rms:.4f}", xy=(0.02, 0.82), xycoords="axes fraction",
                    fontsize=10, color=ORANGE)

        ax.set_ylim(ymin, ymax); ax.set_xlim(xmin, xmax)
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
        ax.legend(facecolor=PLOT_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9,
                  loc="upper right")
        self.s_fig.tight_layout(pad=0.8)
        self.s_canvas.draw()

    def _eval_x0(self):
        try:
            self._ensure_coeffs()
            x0 = float(self.s_x0.text()); T = self._fval(self.s_T, 4.0); L = T / 2
            n = self._target_n()
            v = reconstruct(np.array([x0]), self._a0, self._an, self._bn, L, n)[0]
            self.s_result.setText(f"{v:.6f}")
        except Exception as e:
            self.s_result.setText(f"Err: {e}")

    def _save_figure(self):
        """Save the current Fourier Series plot to a PNG file."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "fourier_series.png",
            "PNG Image (*.png);;PDF File (*.pdf);;SVG File (*.svg)"
        )
        if path:
            self.s_fig.savefig(path, dpi=150, bbox_inches="tight",
                               facecolor=PLOT_BG)

    def _draw_at_current_n(self):
        """Immediately draw the plot at the current n (no animation) when
        the user edits the n field without pressing Play."""
        if self._anim_timer.isActive():
            return   # let the running animation handle drawing
        try:
            key, T = self._ensure_coeffs()
            n = self._target_n()
            self._draw_frame(key, T, n)
            L = T / 2
            self.series_w.show_series(self._a0, self._an, self._bn, L, min(n, 5))
        except Exception:
            pass

    def _jump_sp(self, mode):
        self._stop_animation()
        self._set_nav("spectrum"); self.stack.setCurrentIndex(1)
        self.sp_mode = mode; self._compute_spectrum()

    # ═══════════════════════════════════════════
    #  PAGE 2 — SPECTRUM
    # ═══════════════════════════════════════════
    def _build_spectrum_page(self):
        page = QWidget()
        vl = QVBoxLayout(page); vl.setContentsMargins(10, 8, 10, 6); vl.setSpacing(6)
        h = QLabel("HARMONIC SPECTRUM  &  COEFFICIENT ANALYSIS")
        h.setStyleSheet(f"font-size:15px;font-weight:bold;color:{TEXT};padding:2px;")
        vl.addWidget(h)

        body = QHBoxLayout(); body.setSpacing(8)
        lw = QWidget(); lw.setFixedWidth(310)
        ll = QVBoxLayout(lw); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)

        g1 = QGroupBox("Waveform"); g1l = QVBoxLayout(g1)
        self.sp_combo = QComboBox()
        self.sp_combo.addItems([w["name"] for w in WAVEFORMS])
        self.sp_combo.currentIndexChanged.connect(self._compute_spectrum)
        g1l.addWidget(self.sp_combo); ll.addWidget(g1)

        g2 = QGroupBox("Spectrum Type"); g2l = QGridLayout(g2)
        self.sp_mode = "amplitude"
        for i, (txt, m) in enumerate([("aₙ","an"),("bₙ","bn"),("Amplitude","amplitude"),("Phase","phase")]):
            b = QPushButton(txt); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, mm=m: (setattr(self, "sp_mode", mm), self._compute_spectrum()))
            g2l.addWidget(b, i//2, i%2)
        ll.addWidget(g2)

        g3 = QGroupBox("Range"); g3l = QVBoxLayout(g3)
        rr = QHBoxLayout(); rr.addWidget(_l("n max"))
        self.sp_nmax = _e("20", 45); rr.addWidget(self.sp_nmax); rr.addStretch(); g3l.addLayout(rr)
        rr2 = QHBoxLayout(); rr2.addWidget(_l("T"))
        self.sp_T = _e("4", 45); rr2.addWidget(self.sp_T); rr2.addStretch(); g3l.addLayout(rr2)
        ll.addWidget(g3)

        go = _ab("▶  Compute Spectrum"); go.clicked.connect(self._compute_spectrum); ll.addWidget(go)

        g4 = QGroupBox("Properties"); g4l = QVBoxLayout(g4)
        self.sp_info = QLabel("…"); self.sp_info.setWordWrap(True)
        self.sp_info.setStyleSheet(f"font-family:'Courier New',monospace;font-size:11px;color:{TEXT};")
        g4l.addWidget(self.sp_info); ll.addWidget(g4); ll.addStretch()
        body.addWidget(lw)

        pf = QFrame(); pf.setObjectName("plot_frame")
        pfl = QVBoxLayout(pf); pfl.setContentsMargins(2, 2, 2, 2)
        self.sp_fig = Figure(figsize=(8, 6), dpi=100, facecolor=PLOT_BG)
        self.sp_ax1 = self.sp_fig.add_subplot(211); init_ax(self.sp_ax1)
        self.sp_ax2 = self.sp_fig.add_subplot(212); init_ax(self.sp_ax2)
        self.sp_canvas = _cv(self.sp_fig, pfl)
        body.addWidget(pf, 1)
        vl.addLayout(body, 1); self.stack.addWidget(page)

    def _compute_spectrum(self):
        try:
            w = WAVEFORMS[self.sp_combo.currentIndex()]; key = w["key"]
            T = self._fval(self.sp_T, w["T"])
            N = max(5, int(self._fval(self.sp_nmax, 20)))
            a0, an, bn = compute_coeffs(key, T, max(N, 60))
            ns = np.arange(1, N+1)
            ax1, ax2 = self.sp_ax1, self.sp_ax2
            ax1.clear(); ax2.clear(); init_ax(ax1); init_ax(ax2)
            m = self.sp_mode
            if   m=="an":        v,yl,tt,bc = an[:N],"aₙ","Cosine Coefficients aₙ",ACCENT
            elif m=="bn":        v,yl,tt,bc = bn[:N],"bₙ","Sine Coefficients bₙ",PURPLE
            elif m=="amplitude": v,yl,tt,bc = np.sqrt(an[:N]**2+bn[:N]**2),"Cₙ","Amplitude Spectrum",GREEN
            else:                v,yl,tt,bc = np.arctan2(bn[:N],an[:N])*180/np.pi,"Phase°","Phase Spectrum",ORANGE
            cols=[bc if abs(vv)>1e-4 else "#181830" for vv in v]
            ax1.bar(ns,v,color=cols,alpha=0.85,width=0.7,edgecolor=GRID)
            ax1.axhline(0,color=GRID,lw=0.5)
            ax1.set_title(tt,color=TEXT,fontweight="bold",fontsize=11)
            ax1.set_xlabel("Harmonic n"); ax1.set_ylabel(yl)
            if abs(a0)>1e-5:
                ax1.annotate(f"DC a₀={a0:.4f}",xy=(0.70,0.90),xycoords="axes fraction",fontsize=9,color=TEAL)
            amp=np.sqrt(an[:N]**2+bn[:N]**2)
            te=2*a0**2+np.sum(amp**2/2)
            ce=(2*a0**2+np.cumsum(amp**2/2))/te*100 if te>0 else np.zeros(N)
            ax2.plot(ns,ce,color=TEAL,lw=2.2,label="Energy %")
            ax2.fill_between(ns,ce,alpha=0.12,color=TEAL)
            ax2.axhline(95,color=ORANGE,lw=1,ls="--",alpha=0.6,label="95%")
            ax2.axhline(99,color=RED,lw=1,ls="--",alpha=0.6,label="99%")
            ax2.set_ylim(0,105); ax2.set_xlabel("Harmonic n"); ax2.set_ylabel("Energy %")
            ax2.set_title("Cumulative Energy",color=TEXT,fontweight="bold",fontsize=10)
            ax2.legend(facecolor=PLOT_BG,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
            self.sp_fig.tight_layout(pad=1.4); self.sp_canvas.draw()
            n95=int(np.searchsorted(ce,95)+1); n99=int(np.searchsorted(ce,99)+1)
            self.sp_info.setText(f"DC a₀ = {a0:.5f}\nMax|aₙ|= {np.max(np.abs(an[:N])):.5f}\n"
                                 f"Max|bₙ|= {np.max(np.abs(bn[:N])):.5f}\n\n95% at n={n95}\n99% at n={n99}")
        except Exception as e: self.sp_info.setText(str(e))

    # ═══════════════════════════════════════════
    #  PAGE 3 — FFT
    # ═══════════════════════════════════════════
    def _build_fft_page(self):
        page = QWidget()
        vl = QVBoxLayout(page); vl.setContentsMargins(10, 8, 10, 6); vl.setSpacing(6)
        h = QLabel("FFT ANALYSIS — TIME  vs  FREQUENCY DOMAIN")
        h.setStyleSheet(f"font-size:15px;font-weight:bold;color:{TEXT};padding:2px;")
        vl.addWidget(h)
        body = QHBoxLayout(); body.setSpacing(8)
        lw = QWidget(); lw.setFixedWidth(310)
        ll = QVBoxLayout(lw); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)

        g1 = QGroupBox("Signal"); g1l = QVBoxLayout(g1)
        self.f_combo = QComboBox()
        self.f_combo.addItems([s["name"] for s in FFT_SIGNALS])
        self.f_combo.currentIndexChanged.connect(self._on_fft)
        g1l.addWidget(self.f_combo)
        self.f_desc = QLabel(FFT_SIGNALS[0].get("desc", "")); self.f_desc.setWordWrap(True)
        g1l.addWidget(self.f_desc); ll.addWidget(g1)

        g2 = QGroupBox("Window"); g2l = QVBoxLayout(g2)
        self.f_win = QComboBox()
        self.f_win.addItems(["Rectangular","Hanning","Hamming","Blackman"])
        g2l.addWidget(self.f_win)
        g2l.addWidget(QLabel("Reduces spectral leakage")); ll.addWidget(g2)

        g3 = QGroupBox("Display"); g3l = QVBoxLayout(g3)
        self.f_db = QCheckBox("dB Scale"); self.f_db.stateChanged.connect(lambda: self._compute_fft())
        g3l.addWidget(self.f_db)
        self.f_phase = QCheckBox("Phase"); self.f_phase.stateChanged.connect(lambda: self._compute_fft())
        g3l.addWidget(self.f_phase); ll.addWidget(g3)

        go = _ab("▶  Run FFT"); go.clicked.connect(self._compute_fft); ll.addWidget(go)
        g4 = QGroupBox("Properties"); g4l = QVBoxLayout(g4)
        self.f_info = QLabel("…"); self.f_info.setWordWrap(True)
        self.f_info.setStyleSheet(f"font-family:'Courier New',monospace;font-size:11px;color:{TEXT};")
        g4l.addWidget(self.f_info); ll.addWidget(g4); ll.addStretch()
        body.addWidget(lw)

        pf = QFrame(); pf.setObjectName("plot_frame")
        pfl = QVBoxLayout(pf); pfl.setContentsMargins(2, 2, 2, 2)
        self.f_fig = Figure(figsize=(8, 6), dpi=100, facecolor=PLOT_BG)
        self.f_axt = self.f_fig.add_subplot(211); init_ax(self.f_axt)
        self.f_axf = self.f_fig.add_subplot(212); init_ax(self.f_axf)
        self.f_canvas = _cv(self.f_fig, pfl)
        body.addWidget(pf, 1)
        vl.addLayout(body, 1); self.stack.addWidget(page)

    def _on_fft(self, i):
        self.f_desc.setText(FFT_SIGNALS[i].get("desc", "")); self._compute_fft()

    def _compute_fft(self):
        try:
            sig = FFT_SIGNALS[self.f_combo.currentIndex()]
            fs, dur = sig["fs"], sig["dur"]
            t, y = get_fft_signal(sig["key"], fs, dur); N = len(y)
            wn = self.f_win.currentText()
            if   "Hanning"  in wn: w = np.hanning(N)
            elif "Hamming"  in wn: w = np.hamming(N)
            elif "Blackman" in wn: w = np.blackman(N)
            else:                  w = np.ones(N)
            yw = y*w; wg = max(np.mean(w), 1e-9)
            Y = np.fft.rfft(yw); fr = np.fft.rfftfreq(N, 1/fs)
            mag = np.abs(Y)*2/(N*wg); mag[0] /= 2
            ph = np.angle(Y, deg=True)

            axt, axf = self.f_axt, self.f_axf
            axt.clear(); axf.clear(); init_ax(axt); init_ax(axf)
            axt.plot(t*1e3, y, color="#5cb8ff", lw=1.5, label="Signal")
            if wn != "Rectangular":
                axt.plot(t*1e3, yw, color=ORANGE, lw=1, ls="--", alpha=0.6, label=f"+ {wn}")
            axt.set_title("Time Domain", color=TEXT, fontweight="bold", fontsize=11)
            axt.set_xlabel("Time (ms)"); axt.set_ylabel("Amplitude")
            axt.legend(facecolor=PLOT_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

            db = self.f_db.isChecked()
            mp = 20*np.log10(np.maximum(mag, 1e-10)) if db else mag
            yl = "dBFS" if db else "Magnitude"
            if self.f_phase.isChecked():
                axf.plot(fr, mp, color=GREEN, lw=1.5, label="Magnitude")
                ax2 = axf.twinx()
                ax2.plot(fr, ph, color=ORANGE, lw=0.8, alpha=0.5, label="Phase")
                ax2.set_ylabel("Phase deg", color=ORANGE); ax2.tick_params(colors=ORANGE)
                ax2.set_facecolor("none")
            else:
                axf.fill_between(fr, mp, alpha=0.18, color=GREEN)
                axf.plot(fr, mp, color=GREEN, lw=1.5, label="FFT")
                th = np.max(mp)*0.15; pks = []
                for i in range(1, len(mp)-1):
                    if mp[i]>mp[i-1] and mp[i]>mp[i+1] and mp[i]>th: pks.append((fr[i], mp[i]))
                pks.sort(key=lambda p: p[1], reverse=True)
                for fx, my in pks[:5]:
                    axf.annotate(f"{fx:.0f} Hz", xy=(fx, my), xytext=(fx+fs*0.012, my*0.96),
                                 fontsize=8, color=ORANGE,
                                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
            axf.set_title("Frequency Domain", color=TEXT, fontweight="bold", fontsize=11)
            axf.set_xlabel("Frequency (Hz)"); axf.set_ylabel(yl)
            axf.legend(facecolor=PLOT_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
            self.f_fig.tight_layout(pad=1.4); self.f_canvas.draw()

            pf_ = fr[np.argmax(mag[1:])+1]
            self.f_info.setText(f"N={N}  fs={fs:.0f} Hz\nDur={dur*1e3:.0f} ms\n"
                                f"Df={fs/N:.2f} Hz\nNyquist={fs/2:.0f} Hz\n\n"
                                f"Peak={pf_:.1f} Hz\nMax={np.max(mag):.4f}\nWindow={wn}")
        except Exception as e: self.f_info.setText(str(e))


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(BG))
    p.setColor(QPalette.WindowText, QColor(TEXT))
    p.setColor(QPalette.Base, QColor(BG_INPUT))
    p.setColor(QPalette.AlternateBase, QColor(BG_CARD))
    p.setColor(QPalette.ToolTipBase, QColor(BG_CARD))
    p.setColor(QPalette.ToolTipText, QColor(TEXT))
    p.setColor(QPalette.Text, QColor(TEXT))
    p.setColor(QPalette.Button, QColor(BG_BTN))
    p.setColor(QPalette.ButtonText, QColor(TEXT))
    p.setColor(QPalette.Highlight, QColor(ACCENT))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(p)
    win = FourierLab(); win.show()
    sys.exit(app.exec())
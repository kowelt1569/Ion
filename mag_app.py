"""
MagnetModel Pro — повний standalone застосунок
Магнітна активація · Ridge Regression · Темний UI з сайдбаром
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("MagnetModel")

# ══════════════════════════════════════════════════════════
#  ПАЛІТРА  —  Зелено-м'ятна гама
# ══════════════════════════════════════════════════════════
C = {
    # ── Фони ────────────────────────────────────────────
    "bg0":      "#0A1A12",   # Найтемніший — смарагдово-чорний
    "bg1":      "#0F2318",   # Панелі / сайдбар
    "bg2":      "#163322",   # Картки, поля вводу
    "bg3":      "#1E4A30",   # Hover стан

    # ── Акцент — м'ята / неонова зелень ─────────────────
    "acc":      "#2EE59D",   # Головний акцент (яскрава м'ята)
    "acc2":     "#1AB87A",   # Приглушений (натиснути)
    "acc3":     "#5FFFC0",   # Hover / glow
    "acc_dim":  "#0D6644",   # Дуже темний акцент (фон виділення)

    # ── Другий акцент — лаймово-жовтий ─────────────────
    "lime":     "#A3FF6F",   # Лайм (підзаголовки, label)
    "lime2":    "#70CC3A",   # Темніший лайм

    # ── Текст ───────────────────────────────────────────
    "txt":      "#E8FFF4",   # Основний (крем-зелений)
    "txt2":     "#7EC8A0",   # Другорядний
    "txt3":     "#3D6B50",   # Приглушений / placeholder

    # ── Статуси ─────────────────────────────────────────
    "ok":       "#2EE59D",   # Успіх (=акцент)
    "warn":     "#FFD166",   # Попередження — жовтий
    "err":      "#FF5C7A",   # Помилка — рожево-червоний
    "info":     "#5BC8FF",   # Інфо — блакитний

    # ── Межі ────────────────────────────────────────────
    "border":   "#1E4A30",
    "border2":  "#2EE59D44", # Напівпрозора акцентна межа

    # ── Рядки таблиці ───────────────────────────────────
    "tr_bg":    "#0C2A1A",   # Тренувальні парні
    "tr_bg2":   "#0A2016",   # Тренувальні непарні
    "pr_bg":    "#0F2E20",   # Прогнозні парні
    "pr_bg2":   "#0D2619",   # Прогнозні непарні
}

FONT_MONO = ("Consolas", 11)
FONT_UI   = ("Segoe UI", 11)
FONT_SM   = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_H1   = ("Segoe UI", 18, "bold")
FONT_H2   = ("Segoe UI", 13, "bold")

# ══════════════════════════════════════════════════════════
#  МАТЕМАТИЧНА МОДЕЛЬ (не змінювати!)
# ══════════════════════════════════════════════════════════

def _design_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Матриця ознак: 6 базисних функцій."""
    return np.column_stack([
        x1 * (x2**2),    # F1
        x1 * x2,         # F2
        x2**2,           # F3
        x1,              # F4
        x2,              # F5
        np.ones_like(x1) # F6
    ])

def ridge_regression(A: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Стабільний Ridge МНК: θ = (AᵀA + αI)⁻¹ Aᵀy."""
    I = np.eye(A.shape[1])
    return np.linalg.inv(A.T @ A + alpha * I) @ A.T @ y

def evaluate_model(x1, x2, y_true, coeffs):
    """Оцінка моделі — R² та RMSE."""
    A        = _design_matrix(x1, x2)
    y_pred   = A @ coeffs
    residuals = y_true - y_pred
    ss_res   = np.sum(residuals**2)
    ss_tot   = np.sum((y_true - np.mean(y_true))**2)
    r2       = 1 - ss_res / ss_tot
    rmse     = np.sqrt(np.mean(residuals**2))
    return y_pred, residuals, r2, rmse

def calculate_Y_physical(x1: float, x2: float, coeffs: np.ndarray) -> float:
    """Прогноз одного значення (оригінальна логіка)."""
    F1, F2, F3, F4, F5, F6 = coeffs
    return (
        F1 * x1 * (x2**2) +
        F2 * x1 * x2 +
        F3 * (x2**2) +
        F4 * x1 +
        F5 * x2 +
        F6
    )

# ══════════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════

MPL_RC = {
    "figure.facecolor":   C["bg0"],
    "axes.facecolor":     C["bg1"],
    "axes.edgecolor":     C["border"],
    "axes.labelcolor":    C["txt2"],
    "axes.grid":          True,
    "grid.color":         C["bg2"],
    "grid.linewidth":     0.7,
    "xtick.color":        C["txt3"],
    "ytick.color":        C["txt3"],
    "text.color":         C["txt"],
    "legend.facecolor":   C["bg1"],
    "legend.edgecolor":   C["border"],
    "lines.linewidth":    2.2,
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        150,
    "savefig.facecolor":  C["bg0"],
}
for k, v in MPL_RC.items():
    try:
        matplotlib.rcParams[k] = v
    except Exception:
        pass

# ══════════════════════════════════════════════════════════
#  ГОЛОВНИЙ КЛАС ЗАСТОСУНКУ
# ══════════════════════════════════════════════════════════

class MagnetApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # ── Стан ────────────────────────────────────────────────────────
        self._coeffs: np.ndarray | None = None
        self._train_rows: list[tuple]   = []   # (x1, x2, y)
        self._pred_rows:  list[tuple]   = []   # (x1, x2, y)
        self._history:    list[dict]    = []

        # ── Вікно ───────────────────────────────────────────────────────
        self.title("MagnetModel Pro  ·  v2.0")
        self.geometry("1280x780")
        self.minsize(1100, 680)
        self.configure(bg=C["bg0"])

        self._configure_ttk_styles()
        self._build_ui()

    # ════════════════════════════════════════════════════════════════════
    #  TTK STYLES
    # ════════════════════════════════════════════════════════════════════

    def _configure_ttk_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        # Treeview
        s.configure("Mag.Treeview",
            background=C["bg1"], foreground=C["txt"],
            fieldbackground=C["bg1"], rowheight=26,
            font=FONT_MONO, borderwidth=0,
        )
        s.configure("Mag.Treeview.Heading",
            background=C["bg2"], foreground=C["acc"],
            font=FONT_BOLD, relief="flat",
        )
        s.map("Mag.Treeview",
            background=[("selected", C["acc2"])],
            foreground=[("selected", C["txt"])],
        )
        s.map("Mag.Treeview.Heading",
            background=[("active", C["bg3"])],
        )

        # Scrollbar
        s.configure("Mag.Vertical.TScrollbar",
            troughcolor=C["bg1"], background=C["bg3"],
            borderwidth=0, relief="flat",
        )

        # Separator
        s.configure("Mag.TSeparator", background=C["border"])

        # Notebook (вкладки)
        s.configure("Mag.TNotebook",
            background=C["bg0"], borderwidth=0,
        )
        s.configure("Mag.TNotebook.Tab",
            background=C["bg1"], foreground=C["txt2"],
            padding=[14, 6], font=FONT_SM,
        )
        s.map("Mag.TNotebook.Tab",
            background=[("selected", C["bg2"])],
            foreground=[("selected", C["acc"])],
        )

    # ════════════════════════════════════════════════════════════════════
    #  ПОБУДОВА UI
    # ════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Titlebar
        self._build_titlebar()

        # Основна зона = sidebar + main
        body = tk.Frame(self, bg=C["bg0"])
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        self._build_main(body)

        # Statusbar
        self._build_statusbar()

    # ── TITLEBAR ─────────────────────────────────────────────────────────

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=C["bg1"], height=54)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Зелена смуга-акцент зліва
        tk.Frame(bar, bg=C["acc"], width=5).pack(side="left", fill="y")

        tk.Label(bar, text=" 🌿  MAGNETMODEL PRO",
                 font=("Segoe UI", 15, "bold"),
                 bg=C["bg1"], fg=C["acc"]).pack(side="left", padx=16)

        tk.Label(bar,
                 text="Ridge Regression  ·  Магнітна активація",
                 font=FONT_SM, bg=C["bg1"], fg=C["txt3"]).pack(side="left", padx=4)

        # Версія праворуч
        tk.Label(bar, text="v2.0",
                 font=("Consolas", 9), bg=C["bg1"], fg=C["acc_dim"]).pack(side="right", padx=8)

        tk.Button(bar, text="✕  Вийти",
                  command=self.destroy,
                  bg="#2A0D14", fg=C["err"], relief="flat",
                  font=FONT_SM, padx=14, pady=5,
                  activebackground=C["err"], activeforeground=C["bg0"],
                  cursor="hand2").pack(side="right", padx=12, pady=10)

    # ── SIDEBAR ───────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["bg1"], width=248)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Вертикальна акцентна смуга
        tk.Frame(sb, bg=C["acc"], width=3).place(x=0, y=0, relheight=1)

        def section(text, icon=""):
            # Відступ + кольорова мітка секції
            spacer = tk.Frame(sb, bg=C["bg1"], height=6)
            spacer.pack(fill="x")

            row = tk.Frame(sb, bg=C["bg1"])
            row.pack(fill="x", padx=(14, 10), pady=(6, 2))

            tk.Label(row, text=f"{icon} {text}".upper(),
                     font=("Segoe UI", 8, "bold"),
                     bg=C["bg1"], fg=C["lime"]).pack(side="left")

            tk.Frame(row, bg=C["border"], height=1).pack(
                side="left", fill="x", expand=True, padx=(6, 0), pady=5)

        def btn(parent, text, cmd, style="sec"):
            styles = {
                # bg, fg, active_bg, active_fg
                "pri": (C["acc"],      C["bg0"],  C["acc3"],  C["bg0"]),
                "sec": (C["bg2"],      C["txt2"], C["bg3"],   C["acc"]),
                "ok":  (C["acc_dim"],  C["acc"],  C["acc2"],  C["bg0"]),
                "del": ("#2A0D14",     C["err"],  C["err"],   C["bg0"]),
                "lim": ("#162A0A",     C["lime"], C["lime2"], C["bg0"]),
            }
            bg, fg, abg, afg = styles[style]
            b = tk.Button(parent, text=text, command=cmd,
                          bg=bg, fg=fg, relief="flat",
                          font=FONT_SM, anchor="w", padx=18, pady=8,
                          activebackground=abg, activeforeground=afg,
                          cursor="hand2")
            b.pack(fill="x", padx=(14, 10), pady=2)
            return b

        def entry_field(label, placeholder):
            tk.Label(sb, text=label, font=("Segoe UI", 9, "bold"),
                     bg=C["bg1"], fg=C["txt2"]).pack(anchor="w", padx=16, pady=(7, 1))
            var = tk.StringVar()
            e = tk.Entry(sb, textvariable=var,
                         bg=C["bg2"], fg=C["txt"],
                         insertbackground=C["acc"],
                         relief="flat", font=FONT_MONO,
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         highlightcolor=C["acc"])
            e.pack(fill="x", padx=(14, 10), pady=(1, 0), ipady=7)

            def _focus_in(ev, v=var, ph=placeholder):
                if v.get() == ph:
                    v.set("")
                    ev.widget.config(fg=C["txt"],
                                     highlightbackground=C["acc"])
            def _focus_out(ev, v=var, ph=placeholder):
                if not v.get():
                    v.set(ph)
                    ev.widget.config(fg=C["txt3"],
                                     highlightbackground=C["border"])

            var.set(placeholder)
            e.config(fg=C["txt3"])
            e.bind("<FocusIn>",  _focus_in)
            e.bind("<FocusOut>", _focus_out)
            return var

        # ── Логотип у верхній частині сайдбара ─────────────────────────
        logo = tk.Frame(sb, bg=C["bg2"])
        logo.pack(fill="x", padx=(14, 10), pady=(14, 6))
        tk.Label(logo, text="🌿  MagnetModel",
                 font=("Segoe UI", 11, "bold"),
                 bg=C["bg2"], fg=C["acc"]).pack(pady=(6, 0))
        tk.Label(logo, text="Ridge Regression Engine",
                 font=("Segoe UI", 8),
                 bg=C["bg2"], fg=C["txt3"]).pack(pady=(0, 6))

        # ── Введення ────────────────────────────────────────────────────
        section("Введення даних", "⬡")
        self._v_x1 = entry_field("X₁  —  Витрата",           "наприклад: 2.5")
        self._v_x2 = entry_field("X₂  —  Напруженість",      "наприклад: 150")
        self._v_y  = entry_field("Y   —  Вихідний параметр",  "для навчання")

        # ── Дані ────────────────────────────────────────────────────────
        section("Дані", "◈")
        btn(sb, "➕   Додати рядок",     self._add_row,    "pri")
        btn(sb, "📂   Завантажити файл", self._load_file,  "sec")
        btn(sb, "✖   Видалити рядок",   self._delete_row, "sec")
        btn(sb, "🗑   Очистити все",     self._clear_all,  "del")

        # ── Модель ──────────────────────────────────────────────────────
        section("Модель", "⚙")
        btn(sb, "⚡   Навчити модель",   self._train,      "pri")

        # ── Прогноз ─────────────────────────────────────────────────────
        section("Прогноз", "◎")
        btn(sb, "🔮   Зробити прогноз",  self._predict,    "lim")
        btn(sb, "💾   Зберегти у Excel", self._save,       "ok")

        # ── Графіки ─────────────────────────────────────────────────────
        section("Графіки", "▣")
        btn(sb, "📈   Exp vs Predicted",  self._plot_eval,   "sec")
        btn(sb, "📉   Аналіз залишків",   self._plot_resid,  "sec")
        btn(sb, "📊   Прогнозний графік", self._plot_pred,   "sec")
        btn(sb, "💾   Експорт графіку",   self._export_plot, "sec")

        # ── Нижня кнопка ────────────────────────────────────────────────
        tk.Frame(sb, bg=C["bg1"]).pack(fill="both", expand=True)
        tk.Frame(sb, bg=C["border"], height=1).pack(fill="x", padx=10)
        btn(sb, "🕒   Історія прогнозів", self._show_history, "sec")

    # ── MAIN PANEL ────────────────────────────────────────────────────────

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=C["bg0"])
        main.pack(side="left", fill="both", expand=True)

        # Метрики зверху
        self._build_metric_bar(main)

        # Notebook: Таблиця | Графік
        nb = ttk.Notebook(main, style="Mag.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # Вкладка 1: Таблиця
        tab_data = tk.Frame(nb, bg=C["bg1"])
        nb.add(tab_data, text="  📋  Дані  ")
        self._build_table(tab_data)

        # Вкладка 2: Графік (inline matplotlib)
        tab_plot = tk.Frame(nb, bg=C["bg0"])
        nb.add(tab_plot, text="  📈  Графік  ")
        self._build_inline_plot(tab_plot)

        # Рядок коефіцієнтів
        self._build_coeff_bar(main)

    def _build_metric_bar(self, parent):
        bar = tk.Frame(parent, bg=C["bg0"])
        bar.pack(fill="x", padx=12, pady=(10, 6))

        self._metric_vars = {}
        metrics = [
            ("R²",           "—",   C["acc"],  C["acc_dim"]),
            ("RMSE",         "—",   C["lime"], "#162A0A"),
            ("Навч. точок",  "0",   C["info"], "#0A1E2A"),
            ("Прогнозів",    "0",   C["warn"], "#2A1E0A"),
        ]
        for i, (title, val, color, card_bg) in enumerate(metrics):
            card = tk.Frame(bar, bg=card_bg,
                            highlightthickness=1,
                            highlightbackground=color)
            card.grid(row=0, column=i, padx=6, sticky="ew")
            bar.grid_columnconfigure(i, weight=1)

            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                     bg=card_bg, fg=color).pack(pady=(8, 0))

            var = tk.StringVar(value=val)
            lbl = tk.Label(card, textvariable=var,
                           font=("Consolas", 22, "bold"),
                           bg=card_bg, fg=color)
            lbl.pack(pady=(0, 8))
            self._metric_vars[title] = (var, lbl, color)

    def _build_table(self, parent):
        cols = ("№", "X₁", "X₂", "Y", "Тип")
        widths = (50, 180, 180, 180, 110)

        frame = tk.Frame(parent, bg=C["bg1"])
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(frame, style="Mag.Vertical.TScrollbar")
        vsb.pack(side="right", fill="y")

        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  style="Mag.Treeview",
                                  yscrollcommand=vsb.set,
                                  selectmode="browse")
        vsb.config(command=self._tree.yview)

        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col,
                               command=lambda c=col: self._sort_col(c))
            self._tree.column(col, anchor="center", width=w, minwidth=50)

        # Теги
        self._tree.tag_configure("train_e", background=C["tr_bg"],  foreground=C["acc"])
        self._tree.tag_configure("train_o", background=C["tr_bg2"], foreground=C["acc"])
        self._tree.tag_configure("pred_e",  background=C["pr_bg"],  foreground=C["lime"])
        self._tree.tag_configure("pred_o",  background=C["pr_bg2"], foreground=C["lime"])

        self._tree.pack(fill="both", expand=True)
        self._sort_asc = {}

    def _build_inline_plot(self, parent):
        """Вбудований matplotlib canvas у вкладці."""
        self._fig = Figure(figsize=(8, 5), facecolor=C["bg0"])
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(C["bg1"])
        self._ax.tick_params(colors=C["txt3"])
        for sp in self._ax.spines.values():
            sp.set_edgecolor(C["border"])
        self._ax.set_title("Тут з'являться графіки після навчання",
                           color=C["txt3"], fontsize=11)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(parent, bg=C["bg1"])
        toolbar_frame.pack(fill="x")
        tb = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        tb.config(bg=C["bg1"])
        tb.update()

    def _build_coeff_bar(self, parent):
        bar = tk.Frame(parent, bg=C["bg2"],
                       highlightthickness=1,
                       highlightbackground=C["acc_dim"])
        bar.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(bar, text="КОЕФІЦІЄНТИ:", font=("Segoe UI", 8, "bold"),
                 bg=C["bg2"], fg=C["lime"]).pack(side="left", padx=10, pady=7)

        self._coeff_var = tk.StringVar(value="Модель не навчена")
        tk.Label(bar, textvariable=self._coeff_var,
                 font=("Consolas", 10),
                 bg=C["bg2"], fg=C["acc"]).pack(side="left", padx=6)

    # ── STATUSBAR ─────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=C["bg1"], height=28)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        # Зелена смуга зліва
        tk.Frame(sb, bg=C["acc"], width=3).pack(side="left", fill="y")

        self._status_var  = tk.StringVar(value="● Готово")
        self._status_lbl  = tk.Label(sb, textvariable=self._status_var,
                                     font=FONT_SM, bg=C["bg1"], fg=C["txt2"],
                                     anchor="w")
        self._status_lbl.pack(side="left", padx=10)

        self._rows_var = tk.StringVar(value="Рядків: 0")
        tk.Label(sb, textvariable=self._rows_var,
                 font=("Consolas", 9), bg=C["bg1"], fg=C["acc_dim"]).pack(side="right", padx=12)

    # ════════════════════════════════════════════════════════════════════
    #  TOAST
    # ════════════════════════════════════════════════════════════════════

    def _toast(self, msg: str, level: str = "info", ms: int = 3200):
        colors = {"info": C["info"], "success": C["ok"],
                  "warning": C["warn"], "error": C["err"]}
        icons  = {"info": "◆", "success": "✔", "warning": "⚠", "error": "✖"}
        col    = colors.get(level, C["info"])
        icon   = icons.get(level, "●")

        t = tk.Toplevel(self)
        t.overrideredirect(True)
        t.attributes("-topmost", True)
        t.configure(bg=C["bg1"])

        w, h = 330, 72
        sw   = self.winfo_screenwidth()
        sh   = self.winfo_screenheight()
        t.geometry(f"{w}x{h}+{sw-w-20}+{sh-h-60}")

        tk.Frame(t, bg=col, width=4).pack(side="left", fill="y")
        body = tk.Frame(t, bg=C["bg1"])
        body.pack(fill="both", expand=True, padx=10, pady=8)
        tk.Label(body, text=f"{icon}  {level.upper()}", bg=C["bg1"], fg=col,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(anchor="w")
        tk.Label(body, text=msg, bg=C["bg1"], fg=C["txt"],
                 font=FONT_SM, anchor="w",
                 wraplength=270, justify="left").pack(anchor="w")

        self.after(ms, lambda: t.destroy() if t.winfo_exists() else None)

    # ════════════════════════════════════════════════════════════════════
    #  СТАТУС
    # ════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str, level: str = "info"):
        icons = {"info": "◆", "success": "✔", "warning": "⚠", "error": "✖"}
        cols  = {"info": C["txt2"], "success": C["ok"],
                 "warning": C["warn"], "error": C["err"]}
        self._status_var.set(f"{icons.get(level,'●')}  {msg}")
        self._status_lbl.config(fg=cols.get(level, C["txt2"]))

    def _set_metric(self, key: str, val: str, color: str | None = None):
        var, lbl, default_col = self._metric_vars[key]
        var.set(val)
        if color:
            lbl.config(fg=color)

    def _update_counters(self):
        n_train = len(self._train_rows)
        n_pred  = len(self._pred_rows)
        total   = n_train + n_pred
        self._set_metric("Навч. точок", str(n_train))
        self._set_metric("Прогнозів",   str(n_pred))
        self._rows_var.set(f"Рядків: {total}")

    # ════════════════════════════════════════════════════════════════════
    #  ТАБЛИЦЯ — допоміжні методи
    # ════════════════════════════════════════════════════════════════════

    def _insert_tree_row(self, x1, x2, y, rtype):
        idx   = len(self._tree.get_children()) + 1
        par   = "e" if idx % 2 == 0 else "o"
        tag   = f"{rtype}_{par}"
        label = "Навчання" if rtype == "train" else "Прогноз"
        self._tree.insert("", "end",
                          values=(idx, f"{x1:.4f}", f"{x2:.4f}", f"{y:.4f}", label),
                          tags=(tag, rtype))

    def _reindex(self):
        for i, item in enumerate(self._tree.get_children(), 1):
            vals = list(self._tree.item(item, "values"))
            vals[0] = i
            self._tree.item(item, values=vals)

    def _sort_col(self, col):
        asc  = not self._sort_asc.get(col, True)
        self._sort_asc[col] = asc
        items = [(self._tree.set(k, col), k) for k in self._tree.get_children()]
        try:
            items.sort(key=lambda t: float(t[0].replace(",", ".")), reverse=not asc)
        except ValueError:
            items.sort(key=lambda t: t[0], reverse=not asc)
        for i, (_, k) in enumerate(items):
            self._tree.move(k, "", i)
        self._reindex()

    def _get_entry(self, var: tk.StringVar) -> float:
        """Прочитати float з поля вводу, ігноруючи placeholder."""
        raw = var.get().strip()
        placeholders = {"наприклад: 2.5", "наприклад: 150", "для навчання"}
        if raw in placeholders or not raw:
            raise ValueError("Порожнє поле")
        return float(raw.replace(",", "."))

    def _clear_entries(self):
        placeholders = {
            self._v_x1: "наприклад: 2.5",
            self._v_x2: "наприклад: 150",
            self._v_y:  "для навчання",
        }
        for var, ph in placeholders.items():
            var.set(ph)

    # ════════════════════════════════════════════════════════════════════
    #  ОБРОБНИКИ КНОПОК
    # ════════════════════════════════════════════════════════════════════

    def _add_row(self):
        try:
            x1 = self._get_entry(self._v_x1)
            x2 = self._get_entry(self._v_x2)
            y  = self._get_entry(self._v_y)
        except ValueError as e:
            self._toast(f"Введіть коректні числові значення X₁, X₂ та Y", "error")
            self._set_status("Помилка вводу", "error")
            return

        self._train_rows.append((x1, x2, y))
        self._insert_tree_row(x1, x2, y, "train")
        self._update_counters()
        self._set_status(f"Додано: X₁={x1:.4f}  X₂={x2:.4f}  Y={y:.4f}", "success")
        self._clear_entries()

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Відкрити дані",
            filetypes=[("Excel", "*.xlsx *.xls"), ("CSV", "*.csv"), ("Всі", "*.*")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            self._toast(f"Помилка читання: {e}", "error")
            return

        count = 0
        for _, row in df.iterrows():
            try:
                x1, x2, y = float(row.iloc[0]), float(row.iloc[1]), float(row.iloc[2])
                self._train_rows.append((x1, x2, y))
                self._insert_tree_row(x1, x2, y, "train")
                count += 1
            except Exception:
                continue

        self._update_counters()
        self._toast(f"Завантажено {count} рядків", "success")
        self._set_status(f"Файл завантажено: {os.path.basename(path)}", "success")

    def _delete_row(self):
        sel = self._tree.selection()
        if not sel:
            self._toast("Виберіть рядок для видалення", "warning")
            return
        item = sel[0]
        tags = self._tree.item(item, "tags")
        vals = self._tree.item(item, "values")

        # Видалити з внутрішніх списків
        try:
            r = (float(vals[1]), float(vals[2]), float(vals[3]))
            if "train" in tags and r in self._train_rows:
                self._train_rows.remove(r)
            elif "predict" in tags and r in self._pred_rows:
                self._pred_rows.remove(r)
        except Exception:
            pass

        self._tree.delete(item)
        self._reindex()
        self._update_counters()
        self._set_status("Рядок видалено", "info")

    def _clear_all(self):
        if not messagebox.askyesno("Підтвердження",
                "Очистити всі дані?\nМодель буде скинута."):
            return
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._train_rows.clear()
        self._pred_rows.clear()
        self._coeffs = None
        self._update_counters()
        self._set_metric("R²",   "—", C["acc"])
        self._set_metric("RMSE", "—", C["lime"])
        self._coeff_var.set("Модель не навчена")
        self._set_status("Дані очищено", "info")

    def _train(self):
        if len(self._train_rows) < 6:
            self._toast(
                f"Потрібно ≥6 тренувальних точок (зараз: {len(self._train_rows)})",
                "warning"
            )
            return

        self._set_status("Навчання...", "info")

        def _do():
            x1 = np.array([r[0] for r in self._train_rows])
            x2 = np.array([r[1] for r in self._train_rows])
            y  = np.array([r[2] for r in self._train_rows])
            A  = _design_matrix(x1, x2)
            c  = ridge_regression(A, y, alpha=0.01)
            yp, res, r2, rmse = evaluate_model(x1, x2, y, c)
            self.after(0, lambda: self._train_done(c, r2, rmse, x1, x2, y, yp, res))

        threading.Thread(target=_do, daemon=True).start()

    def _train_done(self, coeffs, r2, rmse, x1, x2, y, y_pred, residuals):
        self._coeffs = coeffs

        r2_col = C["ok"] if r2 > 0.9 else (C["warn"] if r2 > 0.7 else C["err"])
        self._set_metric("R²",   f"{r2:.4f}", r2_col)
        self._set_metric("RMSE", f"{rmse:.4f}", C["lime"])

        c = coeffs
        self._coeff_var.set(
            f"F₁={c[0]:.4f}  F₂={c[1]:.4f}  F₃={c[2]:.4f}"
            f"  F₄={c[3]:.4f}  F₅={c[4]:.4f}  F₆={c[5]:.4f}"
        )

        # Inline графік: Exp vs Predicted
        self._draw_inline_eval(y, y_pred)

        self._toast(f"Навчання завершено  R²={r2:.4f}  RMSE={rmse:.4f}", "success")
        self._set_status(f"Модель навчена | R²={r2:.4f} | RMSE={rmse:.4f}", "success")
        log.info("Model trained: R²=%.4f, RMSE=%.4f", r2, rmse)

    def _predict(self):
        if self._coeffs is None:
            self._toast("Спочатку навчіть модель!", "warning")
            return
        try:
            x1 = self._get_entry(self._v_x1)
            x2 = self._get_entry(self._v_x2)
        except ValueError:
            self._toast("Введіть числові значення X₁ та X₂", "error")
            return

        y = calculate_Y_physical(x1, x2, self._coeffs)
        self._pred_rows.append((x1, x2, y))
        self._history.append({"x1": x1, "x2": x2, "y": round(y, 4)})
        self._insert_tree_row(x1, x2, y, "predict")
        self._update_counters()
        self._set_status(f"Прогноз: Y = {y:.4f}  (X₁={x1:.4f}, X₂={x2:.4f})", "success")
        self._toast(f"Y = {y:.4f}", "success")
        self._clear_entries()

    def _save(self):
        if not self._pred_rows:
            self._toast("Немає прогнозів для збереження", "warning")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="mag_predictions.xlsx",
        )
        if not path:
            return

        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Прогнози"

            # Заголовки
            headers = ["X1", "X2", "Y_predicted"]
            hdr_fill = PatternFill("solid", fgColor="2EE59D")
            hdr_font = Font(bold=True, color="0A1A12")
            thin = Side(style="thin", color="1E4A30")
            brd  = Border(left=thin, right=thin, top=thin, bottom=thin)

            for ci, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=ci, value=h)
                cell.fill = hdr_fill
                cell.font = hdr_font
                cell.alignment = Alignment(horizontal="center")
                cell.border = brd
                ws.column_dimensions[cell.column_letter].width = 18

            for ri, (x1, x2, y) in enumerate(self._pred_rows, 2):
                for ci, v in enumerate([x1, x2, round(y, 4)], 1):
                    cell = ws.cell(row=ri, column=ci, value=v)
                    cell.alignment = Alignment(horizontal="center")
                    cell.border = brd

            wb.save(path)
            self._toast(f"Збережено: {os.path.basename(path)}", "success")
            self._set_status(f"Прогнози збережено → {path}", "success")
        except Exception as e:
            self._toast(f"Помилка збереження: {e}", "error")
            log.error("Save error: %s", e)

    # ════════════════════════════════════════════════════════════════════
    #  ГРАФІКИ
    # ════════════════════════════════════════════════════════════════════

    def _draw_inline_eval(self, y, y_pred):
        """Малює Exp vs Pred у вбудованому canvas."""
        self._ax.clear()
        self._ax.set_facecolor(C["bg1"])
        self._ax.scatter(y, y_pred, color=C["acc"], s=70,
                         zorder=5, alpha=0.9, edgecolors=C["acc3"],
                         linewidths=0.8, label="Дані")
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        self._ax.plot([mn, mx], [mn, mx], color=C["lime"],
                      lw=2, ls="--", label="Ідеал")
        self._ax.set_xlabel("Експериментальні Y")
        self._ax.set_ylabel("Модельні Y")
        self._ax.set_title("Адекватність моделі  (Exp vs Predicted)", color=C["txt"])
        self._ax.legend()
        self._ax.tick_params(colors=C["txt3"])
        for sp in self._ax.spines.values():
            sp.set_edgecolor(C["border"])
        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_eval(self):
        if self._coeffs is None or not self._train_rows:
            self._toast("Навчіть модель перед побудовою графіків", "warning")
            return
        x1 = np.array([r[0] for r in self._train_rows])
        x2 = np.array([r[1] for r in self._train_rows])
        y  = np.array([r[2] for r in self._train_rows])
        y_pred, _, r2, rmse = evaluate_model(x1, x2, y, self._coeffs)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y, y_pred, color=C["acc"], s=80, zorder=5, alpha=0.9,
                   edgecolors=C["acc3"], linewidths=0.8, label="Точки")
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color=C["lime"], lw=2, ls="--", label="Ідеал y=x")
        ax.set_xlabel("Експериментальні Y")
        ax.set_ylabel("Модельні Y")
        ax.set_title(f"Адекватність моделі  —  R²={r2:.4f}  RMSE={rmse:.4f}")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def _plot_resid(self):
        if self._coeffs is None or not self._train_rows:
            self._toast("Навчіть модель перед побудовою графіків", "warning")
            return
        x1 = np.array([r[0] for r in self._train_rows])
        x2 = np.array([r[1] for r in self._train_rows])
        y  = np.array([r[2] for r in self._train_rows])
        y_pred, residuals, _, _ = evaluate_model(x1, x2, y, self._coeffs)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred, residuals, color=C["acc"], s=80,
                   alpha=0.9, edgecolors=C["acc3"], linewidths=0.8)
        ax.axhline(0, color=C["lime"], lw=2, ls="--")
        ax.set_xlabel("Прогнозовані Y")
        ax.set_ylabel("Залишки (Y_exp − Y_pred)")
        ax.set_title("Аналіз залишків")
        fig.tight_layout()
        plt.show()

    def _plot_pred(self):
        if not self._pred_rows:
            self._toast("Немає прогнозів для графіку", "warning")
            return
        df = pd.DataFrame(self._pred_rows, columns=["X1","X2","Y"])

        palette = [C["acc"], C["lime"], C["info"], C["warn"], C["err"], C["acc3"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        for ci, x1_val in enumerate(sorted(df["X1"].unique())):
            sub = df[df["X1"] == x1_val].sort_values("X2")
            color = palette[ci % len(palette)]
            ax.plot(sub["X2"], sub["Y"], marker="o", color=color,
                    linewidth=2.2, markersize=8,
                    markerfacecolor=C["bg1"], markeredgewidth=2,
                    label=f"X₁={x1_val:.3f}")
        ax.set_xlabel("X₂")
        ax.set_ylabel("Y  (прогноз)")
        ax.set_title("Прогнозні значення моделі")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def _export_plot(self):
        if self._coeffs is None or not self._train_rows:
            self._toast("Навчіть модель перед експортом", "warning")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="mag_model_plot.png",
        )
        if not path:
            return
        self._fig.savefig(path, dpi=200, bbox_inches="tight")
        self._toast(f"Графік збережено: {os.path.basename(path)}", "success")

    # ════════════════════════════════════════════════════════════════════
    #  ІСТОРІЯ
    # ════════════════════════════════════════════════════════════════════

    def _show_history(self):
        if not self._history:
            self._toast("Історія порожня", "info")
            return

        win = tk.Toplevel(self)
        win.title("Історія прогнозів")
        win.geometry("520x420")
        win.configure(bg=C["bg0"])
        win.grab_set()

        tk.Label(win, text="ІСТОРІЯ ПРОГНОЗІВ",
                 font=FONT_H2, bg=C["bg0"], fg=C["acc"]).pack(pady=14)

        cols = ("№", "X₁", "X₂", "Y")
        tree = ttk.Treeview(win, columns=cols, show="headings",
                            style="Mag.Treeview")
        for col, w in zip(cols, (60, 140, 140, 140)):
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=w)
        tree.tag_configure("e", background=C["pr_bg"],  foreground=C["lime"])
        tree.tag_configure("o", background=C["pr_bg2"], foreground=C["lime"])

        for i, h in enumerate(self._history, 1):
            tag = "e" if i % 2 == 0 else "o"
            tree.insert("", "end",
                        values=(i, h["x1"], h["x2"], h["y"]),
                        tags=(tag,))
        tree.pack(fill="both", expand=True, padx=12, pady=6)

        tk.Button(win, text="Закрити", command=win.destroy,
                  bg=C["bg2"], fg=C["txt"], relief="flat",
                  font=FONT_SM, padx=20, pady=7,
                  cursor="hand2").pack(pady=12)


# ══════════════════════════════════════════════════════════
#  ТОЧКА ВХОДУ
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = MagnetApp()
    app.mainloop()

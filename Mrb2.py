# -*- coding: utf-8 -*-
"""
PyQt5 + Plotly multi-set plotter with Excel-style copy/paste and tolerance labels.

pip install PyQt5 PyQtWebEngine pandas plotly
"""

import sys, io, csv, os, tempfile  # CHANGED
from pathlib import Path  # NEW
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QMimeData, QUrl  # CHANGED: QUrl
from PyQt5.QtGui import QKeySequence, QIcon, QCloseEvent, QGuiApplication, QClipboard
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QSplitter, QShortcut, QMessageBox, QCheckBox
)
from PyQt5.QtWebEngine import QtWebEngine  # NEW
from PyQt5.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
import plotly.io as pio  # NEW


# ---------- Parsing helpers ----------
def parse_tabular_text(text: str) -> List[List[str]]:
    txt = text.strip("\n\r ")
    if not txt:
        return []
    sample = txt[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except Exception:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    reader = csv.reader(io.StringIO(txt), dialect)
    rows = [list(r) for r in reader]
    for r in rows:
        while len(r) and (r[-1] is None or str(r[-1]).strip() == ""):
            r.pop()
    return rows


# ---------- Table with Ctrl+C / Ctrl+V ----------
class CopyPasteTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["X", "Y"])
        self.setRowCount(200)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setStretchLastSection(True)
        self.append_mode = False

        QShortcut(QKeySequence.Copy, self, activated=self.copy_selection)
        QShortcut(QKeySequence.Paste, self, activated=self.paste_clipboard)
        QShortcut(QKeySequence("Ctrl+Delete"), self, activated=self.clear_selection)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.addAction(QtWidgets.QAction("Copy", self, triggered=self.copy_selection))
        self.addAction(QtWidgets.QAction("Paste", self, triggered=self.paste_clipboard))
        self.addAction(QtWidgets.QAction("Cut", self, triggered=self.cut_selection))
        self.addAction(QtWidgets.QAction("Clear selection", self, triggered=self.clear_selection))
        self.addAction(QtWidgets.QAction("Clear table", self, triggered=self.clear_all))

    def set_append_mode(self, state: bool):
        self.append_mode = state

    def _first_empty_row(self) -> int:
        for r in range(self.rowCount()):
            itx = self.item(r, 0)
            ity = self.item(r, 1)
            if (itx is None or itx.text().strip() == "") and (ity is None or ity.text().strip() == ""):
                return r
        return self.rowCount()

    def clear_selection(self):
        for it in self.selectedItems():
            it.setText("")
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                idx = self.model().index(r, c)
                if self.isItemSelected(idx) and self.item(r, c) is None:
                    self.setItem(r, c, QTableWidgetItem(""))

    def clear_all(self):
        self.setRowCount(0)
        self.setRowCount(200)

    def cut_selection(self):
        self.copy_selection()
        self.clear_selection()

    def copy_selection(self):
        ranges = self.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        rows = []
        for i in range(r.topRow(), r.bottomRow() + 1):
            row = []
            for j in range(r.leftColumn(), r.rightColumn() + 1):
                item = self.item(i, j)
                row.append("" if item is None else item.text())
            rows.append(row)
        tsv = "\n".join("\t".join(map(str, row)) for row in rows)
        s = io.StringIO()
        writer = csv.writer(s, lineterminator="\n")
        for row in rows:
            writer.writerow(row)
        csv_text = s.getvalue()
        mime = QMimeData()
        mime.setText(tsv)
        mime.setData("text/csv", csv_text.encode("utf-8"))
        QGuiApplication.clipboard().setMimeData(mime, QClipboard.Mode.Clipboard)

    def paste_clipboard(self):
        md = QGuiApplication.clipboard().mimeData(QClipboard.Mode.Clipboard)
        text = ""
        if md.hasFormat("text/csv"):
            try:
                text = bytes(md.data("text/csv")).decode("utf-8")
            except Exception:
                text = md.text()
        else:
            text = md.text()
        rows = parse_tabular_text(text)
        if not rows:
            return
        if len(rows[0]) == 1:
            rows = [[str(i), rows[i][0]] for i in range(len(rows))]
        else:
            rows = [row[:2] + ([""] * max(0, 2 - len(row))) for row in rows]
        if self.append_mode:
            start_r = self._first_empty_row()
            start_c = 0
        else:
            anchor = self.currentIndex()
            start_r = max(0, anchor.row()) if anchor.isValid() else 0
            start_c = max(0, anchor.column()) if anchor.isValid() else 0
        needed_rows = start_r + len(rows)
        if needed_rows > self.rowCount():
            self.setRowCount(needed_rows)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                r = start_r + i
                c = min(start_c + j, 1)
                item = self.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(r, c, item)
                item.setText(str(val))


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 + Plotly Multi-Set Plotter")
        self.resize(1280, 780)

        self.sets: Dict[int, pd.DataFrame] = {}
        self.roles: Dict[str, Optional[int]] = {"nominal": None, "min_tol": None, "max_tol": None}
        self._active_set_id: int = 1
        self._tmp_html_path: Optional[str] = None  # NEW

        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(300)
        self._edit_timer.timeout.connect(self._save_table_to_model)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.addWidget(QLabel("Number of sets:"))
        self.spin_sets = QSpinBox()
        self.spin_sets.setRange(2, 99)
        self.spin_sets.setValue(3)
        self.spin_sets.valueChanged.connect(self.on_sets_changed)
        top_bar.addWidget(self.spin_sets)

        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Active set:"))
        self.combo_active = QComboBox()
        self.combo_active.currentIndexChanged.connect(self.on_active_changed)
        top_bar.addWidget(self.combo_active)

        top_bar.addSpacing(12)
        self.chk_append = QCheckBox("Append rows on paste")
        self.chk_append.stateChanged.connect(lambda s: self.table.set_append_mode(bool(s)))
        top_bar.addWidget(self.chk_append)

        top_bar.addStretch(1)
        self.btn_paste = QPushButton("Paste (Ctrl+V)")
        self.btn_paste.clicked.connect(self.on_paste)
        top_bar.addWidget(self.btn_paste)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.on_clear)
        top_bar.addWidget(self.btn_clear)
        self.btn_import = QPushButton("Import CSV")
        self.btn_import.clicked.connect(self.on_import_csv)
        top_bar.addWidget(self.btn_import)
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.on_export_csv)
        top_bar.addWidget(self.btn_export)
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.render_plot)
        top_bar.addWidget(self.btn_plot)

        self.table = CopyPasteTable()
        self.table.itemChanged.connect(self._on_table_edited)

        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.addLayout(top_bar)
        left_v.addWidget(self.table)

        self.web = QWebEngineView()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.web)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        roles_bar = QHBoxLayout()
        roles_bar.setSpacing(12)
        roles_bar.addWidget(QLabel("Nominal set:"))
        self.combo_nominal = QComboBox()
        self.combo_nominal.currentIndexChanged.connect(lambda _: self._on_role_changed("nominal", self.combo_nominal))
        roles_bar.addWidget(self.combo_nominal)
        roles_bar.addWidget(QLabel("Min tol set:"))
        self.combo_min = QComboBox()
        self.combo_min.currentIndexChanged.connect(lambda _: self._on_role_changed("min_tol", self.combo_min))
        roles_bar.addWidget(self.combo_min)
        roles_bar.addWidget(QLabel("Max tol set:"))
        self.combo_max = QComboBox()
        self.combo_max.currentIndexChanged.connect(lambda _: self._on_role_changed("max_tol", self.combo_max))
        roles_bar.addWidget(self.combo_max)
        roles_bar.addStretch(1)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter, 1)
        root.addLayout(roles_bar)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        self._init_sets(self.spin_sets.value())
        self._rebuild_set_selectors()
        self.combo_active.setCurrentIndex(0)
        self.render_plot()

    # ----- Model <-> UI -----
    def _init_sets(self, n: int):
        self.sets.clear()
        for i in range(1, n + 1):
            self.sets[i] = pd.DataFrame(columns=["X", "Y"])
        self.roles = {"nominal": 1 if n >= 1 else None,
                      "min_tol": 2 if n >= 2 else None,
                      "max_tol": 3 if n >= 3 else None}

    def _rebuild_set_selectors(self):
        ids = [str(i) for i in sorted(self.sets.keys())]
        def refill(combo: QComboBox, allow_none: bool):
            old = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if allow_none:
                combo.addItem("None")
            combo.addItems(ids)
            combo.blockSignals(False)
            idx = combo.findText(old)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.combo_active.blockSignals(True)
        self.combo_active.clear()
        self.combo_active.addItems(ids)
        self.combo_active.blockSignals(False)
        refill(self.combo_nominal, True)
        refill(self.combo_min, True)
        refill(self.combo_max, True)
        self._apply_role_display_state()

    def _apply_role_display_state(self):
        def set_combo(combo: QComboBox, role_key: str):
            want = self.roles[role_key]
            target_text = "None" if want is None else str(want)
            idx = combo.findText(target_text)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        set_combo(self.combo_nominal, "nominal")
        set_combo(self.combo_min, "min_tol")
        set_combo(self.combo_max, "max_tol")

    def _on_role_changed(self, role_key: str, combo: QComboBox):
        text = combo.currentText()
        value = None if text == "None" else int(text)
        for k in list(self.roles.keys()):
            if k != role_key and self.roles[k] == value and value is not None:
                self.roles[k] = None
        self.roles[role_key] = value
        self._apply_role_display_state()
        self.render_plot()

    def on_sets_changed(self, n: int):
        self._save_table_to_model()
        current_ids = sorted(self.sets.keys())
        if n > len(current_ids):
            for i in range(len(current_ids) + 1, n + 1):
                self.sets[i] = pd.DataFrame(columns=["X", "Y"])
        elif n < len(current_ids):
            for i in range(len(current_ids), n, -1):
                self.sets.pop(i, None)
                for k in self.roles:
                    if self.roles[k] == i:
                        self.roles[k] = None
        self._rebuild_set_selectors()
        self._active_set_id = min(self._active_set_id, n)
        self.combo_active.setCurrentIndex(self._active_set_id - 1)
        self.statusBar().showMessage(f"Number of sets: {n}", 2500)

    def on_active_changed(self, index: int):
        self._save_table_to_model()
        self._active_set_id = index + 1
        self._load_model_to_table(self._active_set_id)
        self.statusBar().showMessage(f"Active set: {self._active_set_id}", 2000)

    def _on_table_edited(self, *_):
        self._edit_timer.start()

    def _save_table_to_model(self):
        xs, ys = [], []
        for r in range(self.table.rowCount()):
            x_item = self.table.item(r, 0)
            y_item = self.table.item(r, 1)
            x = (x_item.text().strip() if x_item else "")
            y = (y_item.text().strip() if y_item else "")
            if x == "" and y == "":
                continue
            xs.append(x)
            ys.append(y)
        df = pd.DataFrame({"X": xs, "Y": ys})
        self.sets[self._active_set_id] = df

    def _load_model_to_table(self, set_id: int):
        self.table.blockSignals(True)
        self.table.clearContents()
        df = self.sets.get(set_id, pd.DataFrame(columns=["X", "Y"]))
        rows = max(200, len(df))
        self.table.setRowCount(rows)
        for i in range(len(df)):
            self.table.setItem(i, 0, QTableWidgetItem(str(df.iloc[i, 0])))
            self.table.setItem(i, 1, QTableWidgetItem(str(df.iloc[i, 1])))
        self.table.blockSignals(False)

    # ----- Data prep -----
    @staticmethod
    def _prepare_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["X", "Y"])
        out = pd.DataFrame({
            "X": pd.to_numeric(df["X"], errors="coerce"),
            "Y": pd.to_numeric(df["Y"], errors="coerce"),
        }).dropna(how="any")
        if out.empty:
            return out
        out = out.sort_values("X", kind="mergesort")
        out = out[~out["X"].duplicated(keep="first")]
        return out

    def _add_tolerance_band(self, fig: go.Figure, df_min: pd.DataFrame, df_max: pd.DataFrame):
        if df_min.empty or df_max.empty:
            return
        x_min = df_min["X"].to_numpy(); y_min = df_min["Y"].to_numpy()
        x_max = df_max["X"].to_numpy(); y_max = df_max["Y"].to_numpy()
        x_union = np.unique(np.concatenate([x_min, x_max]))
        if x_union.size < 2:
            return
        y_min_u = np.interp(x_union, x_min, y_min)
        y_max_u = np.interp(x_union, x_max, y_max)
        lower = np.minimum(y_min_u, y_max_u)
        upper = np.maximum(y_min_u, y_max_u)
        fig.add_trace(go.Scatter(x=x_union, y=lower, mode="lines",
                                 name="Min Tol", line=dict(width=2, dash="dot"),
                                 hovertemplate="x=%{x}<br>y=%{y}<extra>Min Tol</extra>"))
        fig.add_trace(go.Scatter(x=x_union, y=upper, mode="lines",
                                 name="Max Tol", line=dict(width=2, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(0,0,0,0.08)",
                                 hovertemplate="x=%{x}<br>y=%{y}<extra>Max Tol</extra>"))

    # ----- Plotting -----
    def render_plot(self):
        self._save_table_to_model()
        fig = go.Figure()

        nominal_id = self.roles.get("nominal")
        min_id = self.roles.get("min_tol")
        max_id = self.roles.get("max_tol")

        for i in sorted(self.sets.keys()):
            df = self._prepare_numeric_df(self.sets[i])
            if df.empty:
                continue
            name = f"Set {i}"
            line_kw = dict(width=2)
            dash = None
            if nominal_id == i:
                name = "Nominal"
                line_kw["width"] = 3
            elif i == min_id:
                name = "Min Tol"; dash = "dot"
            elif i == max_id:
                name = "Max Tol"; dash = "dot"
            trace = go.Scatter(x=df["X"], y=df["Y"], mode="lines+markers",
                               name=name, line=line_kw, connectgaps=False)
            if dash:
                trace.update(line_dash=dash)
            fig.add_trace(trace)

        if min_id is not None and max_id is not None:
            df_min = self._prepare_numeric_df(self.sets.get(min_id, pd.DataFrame()))
            df_max = self._prepare_numeric_df(self.sets.get(max_id, pd.DataFrame()))
            self._add_tolerance_band(fig, df_min, df_max)

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=10, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="X", yaxis_title="Y",
        )

        # CHANGED: write to a temp HTML file and load via URL (reliable in QWebEngine)
        html = pio.to_html(fig, include_plotlyjs="full", full_html=True)
        self._cleanup_tmp_html()
        f = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
        with f:
            f.write(html)
        self._tmp_html_path = f.name
        self.web.setUrl(QUrl.fromLocalFile(self._tmp_html_path))
        self.statusBar().showMessage("Plotted", 1500)

    # NEW
    def _cleanup_tmp_html(self):
        if self._tmp_html_path and os.path.exists(self._tmp_html_path):
            try:
                os.remove(self._tmp_html_path)
            except Exception:
                pass
        self._tmp_html_path = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_table_to_model()
        self._cleanup_tmp_html()  # NEW
        event.accept()


def main():
    QtWebEngine.initialize()  # NEW: explicit init
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

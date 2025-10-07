# -*- coding: utf-8 -*-
"""
PyQt5 + Plotly multi-set plotter (M1)
- CopyPasteTable supports Ctrl+C / Ctrl+V with Excel/CSV
- Manage N data sets, switch active set, plot all sets
- Bottom bar: label sets as Nominal / Min Tol / Max Tol (styling hooks included)
"""
import sys
import io
import csv
from typing import Dict, Optional, List

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QSplitter, QShortcut, QMessageBox
)
from PyQt5.QtGui import QKeySequence, QIcon, QCloseEvent, QGuiApplication, QClipboard, QMimeData
from PyQt5.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go


# ----------------------------
# Helper: robust text->rows parser
# ----------------------------
def parse_tabular_text(text: str) -> List[List[str]]:
    """Accept TSV/CSV/plain. Auto-detect delimiter. Returns rows (list of list of str)."""
    txt = text.strip("\n\r ")
    if not txt:
        return []
    # Try explicit CSV first (clipboard may set text/csv)
    sample = txt[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except Exception:
        # Fallback heuristic
        if "\t" in sample:
            dialect = csv.excel_tab
        else:
            dialect = csv.excel
    reader = csv.reader(io.StringIO(txt), dialect)
    rows = [list(r) for r in reader]
    # Trim trailing empty columns
    for r in rows:
        while len(r) and (r[-1] is None or str(r[-1]).strip() == ""):
            r.pop()
    return rows


# ----------------------------
# Table with Ctrl+C / Ctrl+V
# ----------------------------
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

        QShortcut(QKeySequence.Copy, self, activated=self.copy_selection)
        QShortcut(QKeySequence.Paste, self, activated=self.paste_clipboard)
        QShortcut(QKeySequence("Ctrl+Delete"), self, activated=self.clear_selection)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        act_copy = QtWidgets.QAction("Copy", self, triggered=self.copy_selection)
        act_paste = QtWidgets.QAction("Paste", self, triggered=self.paste_clipboard)
        act_clear_sel = QtWidgets.QAction("Clear selection", self, triggered=self.clear_selection)
        act_clear_all = QtWidgets.QAction("Clear table", self, triggered=self.clear_all)
        self.addAction(act_copy)
        self.addAction(act_paste)
        self.addAction(act_clear_sel)
        self.addAction(act_clear_all)

    def clear_selection(self):
        for it in self.selectedItems():
            it.setText("")
        # If there were empty cells without QTableWidgetItem, ensure they exist for clearing
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                if self.item(r, c) is None and self.isItemSelected(self.model().index(r, c)):
                    self.setItem(r, c, QTableWidgetItem(""))

    def clear_all(self):
        self.setRowCount(0)
        self.setRowCount(200)

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

        # Build TSV for text/plain and CSV for text/csv
        # TSV text
        tsv = "\n".join("\t".join(map(str, row)) for row in rows)

        # CSV text
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
        cb = QGuiApplication.clipboard()
        md = cb.mimeData(QClipboard.Mode.Clipboard)
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

        # Normalize to 2 columns: if 1 col => Y, auto X=0..n-1; if >2 => take first 2
        if len(rows[0]) == 1:
            rows = [[str(i), rows[i][0]] for i in range(len(rows))]
        else:
            rows = [row[:2] + ([""] * max(0, 2 - len(row))) for row in rows]

        anchor = self.currentIndex()
        start_r = max(0, anchor.row()) if anchor.isValid() else 0
        start_c = max(0, anchor.column()) if anchor.isValid() else 0

        needed_rows = start_r + len(rows)
        if needed_rows > self.rowCount():
            self.setRowCount(needed_rows)

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                r = start_r + i
                c = min(start_c + j, 1)  # restrict to 2 columns
                item = self.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(r, c, item)
                item.setText(str(val))


# ----------------------------
# Main Window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Set Plotter (PyQt5 + Plotly)")
        self.resize(1200, 720)

        # Data structures
        self.sets: Dict[int, pd.DataFrame] = {}
        self.roles: Dict[str, Optional[int]] = {"nominal": None, "min_tol": None, "max_tol": None}
        self._active_set_id: int = 1
        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(300)
        self._edit_timer.timeout.connect(self._save_table_to_model)

        # Top controls
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.addWidget(QLabel("Number of sets:"))
        self.spin_sets = QSpinBox()
        self.spin_sets.setRange(2, 99)
        self.spin_sets.setValue(2)
        self.spin_sets.valueChanged.connect(self.on_sets_changed)
        top_bar.addWidget(self.spin_sets)

        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Active set:"))
        self.combo_active = QComboBox()
        self.combo_active.currentIndexChanged.connect(self.on_active_changed)
        top_bar.addWidget(self.combo_active)

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

        # Left: table
        self.table = CopyPasteTable()
        self.table.itemChanged.connect(self._on_table_edited)

        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.addLayout(top_bar)
        left_v.addWidget(self.table)

        # Right: plot view
        self.web = QWebEngineView()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.web)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Bottom roles bar
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

        # Central layout
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter, 1)
        root.addLayout(roles_bar)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

        # Init sets and UI
        self._init_sets(self.spin_sets.value())
        self._rebuild_set_selectors()
        self.combo_active.setCurrentIndex(0)  # triggers load

        # Initial empty plot
        self.render_plot()

    # ----- Model <-> UI sync -----
    def _init_sets(self, n: int):
        self.sets.clear()
        for i in range(1, n + 1):
            self.sets[i] = pd.DataFrame(columns=["X", "Y"])

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
            # try to restore
            idx = combo.findText(old)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        # Active
        self.combo_active.blockSignals(True)
        self.combo_active.clear()
        self.combo_active.addItems(ids)
        self.combo_active.blockSignals(False)
        # Roles
        refill(self.combo_nominal, True)
        refill(self.combo_min, True)
        refill(self.combo_max, True)
        # Enforce uniqueness display
        self._apply_role_display_state()

    def _apply_role_display_state(self):
        # Ensure uniqueness of roles in mapping
        used = set(v for v in self.roles.values() if v is not None)
        # Update combos to reflect current choices (no disabling for simplicity at M1)
        # Set combo values from roles
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
        # Enforce uniqueness: remove this role from others if conflict
        # Also prevent multiple roles pointing to same set by reassigning previous holder to None
        for k in list(self.roles.keys()):
            if k != role_key and self.roles[k] == value and value is not None:
                self.roles[k] = None
        self.roles[role_key] = value
        self._apply_role_display_state()
        # Re-render to reflect styling hooks
        self.render_plot()

    def on_sets_changed(self, n: int):
        # Save current table before restructuring
        self._save_table_to_model()
        # Resize model
        current_ids = sorted(self.sets.keys())
        if n > len(current_ids):
            for i in range(len(current_ids) + 1, n + 1):
                self.sets[i] = pd.DataFrame(columns=["X", "Y"])
        elif n < len(current_ids):
            # Truncate extra sets (remove from end)
            for i in range(len(current_ids), n, -1):
                self.sets.pop(i, None)
                # Remove roles pointing to removed set
                for k in self.roles:
                    if self.roles[k] == i:
                        self.roles[k] = None
        self._rebuild_set_selectors()
        # Clamp active set id
        self._active_set_id = min(self._active_set_id, n)
        self.combo_active.setCurrentIndex(self._active_set_id - 1)
        self.statusBar().showMessage(f"Number of sets: {n}", 3000)

    def on_active_changed(self, index: int):
        # Save previous active set first
        self._save_table_to_model()
        self._active_set_id = index + 1
        self._load_model_to_table(self._active_set_id)
        self.statusBar().showMessage(f"Active set: {self._active_set_id}", 2000)

    def _on_table_edited(self, *_):
        self._edit_timer.start()

    def _save_table_to_model(self):
        # Extract table to DataFrame for active set
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
            x = str(df.iloc[i, 0]) if i < len(df) else ""
            y = str(df.iloc[i, 1]) if i < len(df) else ""
            self.table.setItem(i, 0, QTableWidgetItem(x))
            self.table.setItem(i, 1, QTableWidgetItem(y))
        self.table.blockSignals(False)

    # ----- Buttons -----
    def on_paste(self):
        self.table.paste_clipboard()
        self._on_table_edited()

    def on_clear(self):
        self.table.clear_all()
        self._on_table_edited()

    def on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv *.txt);;All Files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, header=None)
            if df.shape[1] == 1:
                df = pd.DataFrame({"X": range(len(df)), "Y": df.iloc[:, 0]})
            else:
                df = df.iloc[:, :2]
                df.columns = ["X", "Y"]
            self.sets[self._active_set_id] = df
            self._load_model_to_table(self._active_set_id)
            self.statusBar().showMessage(f"Imported {len(df)} rows into set {self._active_set_id}", 4000)
        except Exception as e:
            QMessageBox.warning(self, "Import error", f"Failed to import:\n{e}")

    def on_export_csv(self):
        self._save_table_to_model()
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"set_{self._active_set_id}.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = self.sets.get(self._active_set_id, pd.DataFrame(columns=["X", "Y"]))
            df.to_csv(path, index=False)
            self.statusBar().showMessage(f"Exported set {self._active_set_id} to {path}", 4000)
        except Exception as e:
            QMessageBox.warning(self, "Export error", f"Failed to export:\n{e}")

    # ----- Plotting -----
    def _prepare_numeric_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["X", "Y"])
        out = pd.DataFrame({
            "X": pd.to_numeric(df["X"], errors="coerce"),
            "Y": pd.to_numeric(df["Y"], errors="coerce"),
        }).dropna(how="any")
        return out

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
                name += " (Nominal)"
                line_kw["width"] = 3
                dash = None
            elif i in {min_id, max_id}:
                name += " (Tol)"
                dash = "dash"

            fig.add_trace(go.Scatter(
                x=df["X"], y=df["Y"],
                mode="lines+markers",
                name=name,
                line=line_kw,
                connectgaps=False
            ).update(line_dash=dash))

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=10, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="X",
            yaxis_title="Y",
        )

        html = fig.to_html(include_plotlyjs="full")
        self.web.setHtml(html)
        self.statusBar().showMessage("Plotted", 1500)

    # ----- Close handling -----
    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_table_to_model()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

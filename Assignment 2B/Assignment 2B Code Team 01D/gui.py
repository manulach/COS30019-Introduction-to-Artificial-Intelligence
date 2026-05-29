"""
COS30019 Assignment 2B: Traffic-Based Route Guidance System
Graphical User Interface

Tkinter-based GUI providing:
  - Origin / Destination SCATS site selection
  - Time-of-day selector (15-minute intervals)
  - ML model selector (LSTM / GRU / CNN_LSTM)
  - Number of routes (k = 1–5)
  - Train models button
  - Find Routes button
  - Results panel with ranked routes and travel times
  - Prediction chart (traffic flow over the day for selected site)
  - Configuration file support (config.json)

Team ID: 01D
Team Members: Saniru, Haresh, Manula

Usage:
    python gui.py
"""

import os
import sys
import json
import pickle
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Project imports
from data_processor import run_pipeline
from route_finder   import build_graph, apply_travel_times, yen_k_shortest
from travel_time    import get_speed_category, flow_per15min_to_vph

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


# ──────────
# CONFIG
# ──────────

DEFAULT_CONFIG = {
    "default_origin":      "2000",
    "default_destination": "3002",
    "default_model":       "LSTM",
    "default_k":           5,
    "default_interval":    36,
    "window_width":        1100,
    "window_height":       750,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ───────────
# HELPERS
# ───────────

def interval_to_time(interval: int) -> str:
    """Convert 0-95 interval index to HH:MM string."""
    total_min = interval * 15
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def time_to_interval(time_str: str) -> int:
    """Convert HH:MM string to 0-95 interval index."""
    h, m = map(int, time_str.split(":"))
    return (h * 60 + m) // 15


def get_site_label(site_id: str, loc_map: dict) -> str:
    return f"{site_id} — {loc_map.get(site_id, '')}"


# ──────────────────
# MAIN APPLICATION
# ──────────────────

class TBRGSApp(tk.Tk):
    """Main application window for the Traffic-Based Route Guidance System."""

    def __init__(self):
        super().__init__()
        self.cfg        = load_config()
        self.site_info  = None
        self.loc_map    = {}
        self.graph      = None
        self.coords     = None
        self.models     = {}          # {"LSTM": keras_model, ...}
        self.scaler     = None
        self.per_site   = {}
        self._load_assets()

        self.title("TBRGS — Traffic-Based Route Guidance System")
        self.geometry(f"{self.cfg['window_width']}x{self.cfg['window_height']}")
        self.resizable(True, True)
        self._build_ui()

    # ── Asset loading ───────────

    def _load_assets(self):
        """Load site info, graph, saved models and scaler (if available)."""
        site_csv = os.path.join(DATA_DIR, "site_info.csv")
        if os.path.exists(site_csv):
            self.site_info = pd.read_csv(site_csv)
            self.site_info["scats_id"] = (
                self.site_info["scats_id"].astype(str).str.zfill(4))
            self.loc_map = dict(
                zip(self.site_info["scats_id"], self.site_info["location"]))
            self.graph, self.coords = build_graph(self.site_info)

        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        per_site_path = os.path.join(DATA_DIR, "per_site_data.pkl")
        if os.path.exists(per_site_path):
            with open(per_site_path, "rb") as f:
                self.per_site = pickle.load(f)

        # Load trained Keras models (silence TF warnings)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        try:
            from tensorflow.keras.models import load_model
            for name in ["LSTM", "GRU", "CNN_LSTM"]:
                path = os.path.join(MODEL_DIR, f"{name}_best.keras")
                if os.path.exists(path):
                    self.models[name] = load_model(path, compile=False)
        except Exception:
            pass

    # ── UI construction ────────────

    def _build_ui(self):
        # ── top bar ─────────────────────
        top = tk.Frame(self, bg="#1a3c5e", pady=8)
        top.pack(fill="x")
        tk.Label(top, text="🚦 TBRGS — Traffic-Based Route Guidance System",
                 font=("Helvetica", 14, "bold"),
                 bg="#1a3c5e", fg="white").pack()

        # ── main pane ────────────────────
        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        left  = tk.Frame(main, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_results(right)

    def _build_controls(self, parent):
        """Left panel: input controls."""
        # ── Data & Training ─────────────────
        grp1 = ttk.LabelFrame(parent, text="Data & Training")
        grp1.pack(fill="x", pady=(0, 6))

        btn_proc = ttk.Button(grp1, text="① Process Data",
                              command=self._on_process_data)
        btn_proc.pack(fill="x", padx=6, pady=3)

        btn_train = ttk.Button(grp1, text="② Train ML Models",
                               command=self._on_train)
        btn_train.pack(fill="x", padx=6, pady=3)

        # Status indicators
        self._data_status  = tk.StringVar(value="⚠ Data not processed")
        self._model_status = tk.StringVar(value="⚠ Models not trained")

        self._update_status_labels()

        tk.Label(grp1, textvariable=self._data_status,
                 font=("Helvetica", 8)).pack(padx=6, pady=1)
        tk.Label(grp1, textvariable=self._model_status,
                 font=("Helvetica", 8)).pack(padx=6, pady=(1, 4))

        # ── Route parameters ─────────────────
        grp2 = ttk.LabelFrame(parent, text="Route Parameters")
        grp2.pack(fill="x", pady=(0, 6))

        site_ids = sorted(self.loc_map.keys()) if self.loc_map else []
        labels   = [get_site_label(s, self.loc_map) for s in site_ids]

        # Origin
        tk.Label(grp2, text="Origin:").pack(anchor="w", padx=6, pady=(4, 0))
        self._origin_var = tk.StringVar()
        origin_cb = ttk.Combobox(grp2, textvariable=self._origin_var,
                                 values=labels, state="readonly", width=36)
        origin_cb.pack(fill="x", padx=6, pady=2)
        self._set_combo_by_id(origin_cb, self.cfg["default_origin"],
                               site_ids, labels)

        # Destination
        tk.Label(grp2, text="Destination:").pack(anchor="w", padx=6,
                                                  pady=(4, 0))
        self._dest_var = tk.StringVar()
        dest_cb = ttk.Combobox(grp2, textvariable=self._dest_var,
                                values=labels, state="readonly", width=36)
        dest_cb.pack(fill="x", padx=6, pady=2)
        self._set_combo_by_id(dest_cb, self.cfg["default_destination"],
                               site_ids, labels)

        # Time of day
        tk.Label(grp2, text="Time of day:").pack(anchor="w", padx=6,
                                                  pady=(4, 0))
        times = [interval_to_time(i) for i in range(96)]
        self._time_var = tk.StringVar(
            value=interval_to_time(self.cfg["default_interval"]))
        time_cb = ttk.Combobox(grp2, textvariable=self._time_var,
                                values=times, state="readonly", width=10)
        time_cb.pack(anchor="w", padx=6, pady=2)

        # ML model selector
        tk.Label(grp2, text="Prediction model:").pack(anchor="w",
                                                       padx=6, pady=(4, 0))
        self._model_var = tk.StringVar(value=self.cfg["default_model"])
        model_cb = ttk.Combobox(grp2, textvariable=self._model_var,
                                 values=["LSTM", "GRU", "CNN_LSTM"],
                                 state="readonly", width=12)
        model_cb.pack(anchor="w", padx=6, pady=2)

        # Number of routes
        tk.Label(grp2, text="Number of routes (k):").pack(
            anchor="w", padx=6, pady=(4, 0))
        self._k_var = tk.IntVar(value=self.cfg["default_k"])
        k_spin = ttk.Spinbox(grp2, from_=1, to=5,
                              textvariable=self._k_var, width=5)
        k_spin.pack(anchor="w", padx=6, pady=(2, 6))

        # ── Find Routes button ─────────────────
        btn_route = ttk.Button(parent, text="🔍  Find Top-k Routes",
                               command=self._on_find_routes)
        btn_route.pack(fill="x", pady=4)

        # ── Progress bar ─────────────────
        self._progress = ttk.Progressbar(parent, mode="indeterminate")
        self._progress.pack(fill="x", pady=2)

    def _build_results(self, parent):
        """Right panel: results + chart."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        # ── Routes tab ───────────────────
        routes_frame = ttk.Frame(notebook)
        notebook.add(routes_frame, text="  Routes  ")

        self._routes_text = scrolledtext.ScrolledText(
            routes_frame, font=("Courier", 10),
            state="disabled", wrap="word")
        self._routes_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._write_routes_text(
            "Select an origin and destination, then click "
            "'Find Top-k Routes'.\n\nMake sure models are trained first.")

        # ── Chart tab ───────────────────
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="  Traffic Chart  ")

        self._fig = Figure(figsize=(7, 3.5), dpi=96)
        self._ax  = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True,
                                           padx=4, pady=4)
        self._draw_empty_chart()

        # ── Model metrics tab ──────────────
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="  Model Metrics  ")

        self._metrics_text = scrolledtext.ScrolledText(
            metrics_frame, font=("Courier", 10),
            state="disabled", wrap="word")
        self._metrics_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._load_metrics_display()

    # ── Event handlers ─────────────

    def _on_process_data(self):
        """Run data pipeline in background thread."""
        def worker():
            self._progress.start()
            try:
                run_pipeline()
                self._load_assets()
                self._update_status_labels()
                messagebox.showinfo("Done", "Data processing complete.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self._progress.stop()
        threading.Thread(target=worker, daemon=True).start()

    def _on_train(self):
        """Train all 3 ML models in background thread."""
        def worker():
            self._progress.start()
            try:
                from train_models import run_training
                results, scaler = run_training()
                self.scaler  = scaler
                self.models  = {
                    name: res["model"] for name, res in results.items()}
                with open(os.path.join(DATA_DIR,
                                        "per_site_data.pkl"), "rb") as f:
                    self.per_site = pickle.load(f)
                self._update_status_labels()
                self._load_metrics_display()
                messagebox.showinfo("Done", "Training complete. "
                                    "See Model Metrics tab.")
            except Exception as e:
                messagebox.showerror("Training error", str(e))
            finally:
                self._progress.stop()
        threading.Thread(target=worker, daemon=True).start()

    def _on_find_routes(self):
        """Find and display top-k routes."""
        if self.graph is None or self.site_info is None:
            messagebox.showwarning("Not ready",
                                   "Run data processing first.")
            return

        origin = self._origin_var.get().split(" — ")[0].strip()
        dest   = self._dest_var.get().split(" — ")[0].strip()
        k      = self._k_var.get()
        time_s = self._time_var.get()
        model_name = self._model_var.get()
        interval   = time_to_interval(time_s)

        if not origin or not dest:
            messagebox.showwarning("Input error",
                                   "Please select origin and destination.")
            return
        if origin == dest:
            messagebox.showwarning("Input error",
                                   "Origin and destination must be different.")
            return

        def worker():
            self._progress.start()
            try:
                flow_lookup = self._build_flow_lookup(
                    model_name, interval)
                timed = apply_travel_times(
                    self.graph, self.coords, flow_lookup)
                routes = yen_k_shortest(
                    timed, self.coords, origin, dest, k=k)

                self.after(0, self._display_routes,
                           routes, origin, dest, time_s, model_name,
                           flow_lookup)
                self.after(0, self._update_chart,
                           origin, model_name, interval)
            except Exception as e:
                self.after(0, messagebox.showerror, "Route error", str(e))
            finally:
                self.after(0, self._progress.stop)
        threading.Thread(target=worker, daemon=True).start()

    # ── Display helpers ──────────────

    def _display_routes(self, routes, origin, dest, time_s,
                         model_name, flow_lookup):
        lines = []
        lines.append("=" * 60)
        lines.append(f"  TBRGS Route Results")
        lines.append("=" * 60)
        lines.append(f"  Origin     : {origin} — {self.loc_map.get(origin,'')}")
        lines.append(f"  Destination: {dest} — {self.loc_map.get(dest,'')}")
        lines.append(f"  Time       : {time_s}")
        lines.append(f"  Model      : {model_name}")
        lines.append("-" * 60)

        if not routes:
            lines.append("\n  ⚠  No route found between these sites.")
        else:
            for r in routes:
                lines.append(
                    f"\n  Route {r['rank']}   ⏱ {r['cost_min']:.1f} min"
                    f"   ({r['hops']} intersections)")
                lines.append(
                    "  Path: " +
                    " → ".join(r["path"]))
                # Show flow/speed at each hop
                for i in range(len(r["path"]) - 1):
                    node = r["path"][i + 1]
                    flow = flow_lookup.get(node, 0.0)
                    vph  = flow_per15min_to_vph(flow)
                    cat  = get_speed_category(vph)
                    lines.append(
                        f"         → {node} ({self.loc_map.get(node,'')})"
                        f"  [{cat}, {int(vph)} veh/hr]")

        lines.append("\n" + "=" * 60)
        self._write_routes_text("\n".join(lines))

    def _write_routes_text(self, text: str):
        self._routes_text.config(state="normal")
        self._routes_text.delete("1.0", "end")
        self._routes_text.insert("end", text)
        self._routes_text.config(state="disabled")

    def _update_chart(self, site_id: str, model_name: str,
                       interval: int):
        """Plot predicted traffic flow over the day for the origin site."""
        self._ax.clear()

        if (not self.models or model_name not in self.models
                or self.scaler is None
                or site_id not in self.per_site):
            self._draw_empty_chart()
            return

        model    = self.models[model_name]
        X_test   = self.per_site[site_id]["X_test"]
        y_actual = self.per_site[site_id]["y_test"]

        # Use up to 96 intervals (one day)
        n = min(96, len(X_test))
        X_in  = X_test[:n].reshape(n, -1, 1)
        y_pred = model.predict(X_in, verbose=0)

        y_pred_inv = self.scaler.inverse_transform(
            y_pred.reshape(-1, 1)).flatten()
        y_true_inv = self.scaler.inverse_transform(
            y_actual[:n].reshape(-1, 1)).flatten()

        times = [interval_to_time(i) for i in range(n)]
        x     = range(n)

        self._ax.plot(x, y_true_inv, label="Actual",
                      color="steelblue", linewidth=1.5)
        self._ax.plot(x, y_pred_inv, label=f"{model_name} prediction",
                      color="darkorange", linewidth=1.5, linestyle="--")
        if interval < n:
            self._ax.axvline(interval, color="red",
                              linestyle=":", label="Selected time")

        tick_idx = list(range(0, n, 8))
        self._ax.set_xticks(tick_idx)
        self._ax.set_xticklabels([times[i] for i in tick_idx],
                                   rotation=45, fontsize=7)
        self._ax.set_title(
            f"Traffic flow — site {site_id}  ({self.loc_map.get(site_id,'')})",
            fontsize=9)
        self._ax.set_ylabel("Vehicles / 15 min")
        self._ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

    def _draw_empty_chart(self):
        self._ax.clear()
        self._ax.text(0.5, 0.5, "Train models and find a route\nto see traffic prediction",
                       ha="center", va="center", transform=self._ax.transAxes,
                       fontsize=10, color="grey")
        self._canvas.draw()

    def _load_metrics_display(self):
        metrics_path = os.path.join(BASE_DIR, "results", "metrics_summary.json")
        if not os.path.exists(metrics_path):
            self._write_metrics("No metrics yet — train models first.")
            return

        with open(metrics_path) as f:
            data = json.load(f)

        lines = ["=" * 55, "  ML Model Evaluation Metrics", "=" * 55,
                 f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Time(s)':>9}",
                 "  " + "-" * 47]
        for name, m in data.items():
            lines.append(
                f"  {name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                f"{m['MAPE']:>8.2f} {m.get('train_time_s', 0):>9.1f}")
        lines += ["=" * 55, "",
                  "  Metrics computed on held-out test set (20%).",
                  "  MAE  : Mean Absolute Error (vehicles/15 min)",
                  "  RMSE : Root Mean Squared Error",
                  "  MAPE : Mean Absolute Percentage Error (%)"]
        self._write_metrics("\n".join(lines))

    def _write_metrics(self, text: str):
        self._metrics_text.config(state="normal")
        self._metrics_text.delete("1.0", "end")
        self._metrics_text.insert("end", text)
        self._metrics_text.config(state="disabled")

    # ── Utility ────────────

    def _build_flow_lookup(self, model_name: str,
                            interval: int) -> dict:
        """
        Build {site_id: flow_15min} for all sites using the
        selected model at the specified time interval.
        Falls back to 0 if model not available.
        """
        site_ids = list(self.coords.keys())
        flow_lookup = {sid: 0.0 for sid in site_ids}

        if (model_name in self.models and self.scaler is not None
                and self.per_site):
            model = self.models[model_name]
            for sid in site_ids:
                if sid not in self.per_site:
                    continue
                X_test = self.per_site[sid]["X_test"]
                idx    = min(interval, len(X_test) - 1)
                x      = X_test[idx].reshape(1, -1, 1)
                pred   = float(model.predict(x, verbose=0)[0][0])
                flow   = float(
                    self.scaler.inverse_transform([[pred]])[0][0])
                flow_lookup[sid] = max(0.0, flow)

        return flow_lookup

    def _update_status_labels(self):
        data_ok  = os.path.exists(os.path.join(DATA_DIR, "X_train.npy"))
        model_ok = any(
            os.path.exists(os.path.join(MODEL_DIR, f"{n}_best.keras"))
            for n in ["LSTM", "GRU", "CNN_LSTM"])

        self._data_status.set(
            "✓ Data processed" if data_ok else "⚠ Data not processed")
        self._model_status.set(
            "✓ Models trained" if model_ok else "⚠ Models not trained")

    @staticmethod
    def _set_combo_by_id(combo, site_id, site_ids, labels):
        if site_id in site_ids:
            idx = site_ids.index(site_id)
            combo.current(idx)
        elif labels:
            combo.current(0)


# ────────────────
# ENTRY POINT
# ────────────────

def main():
    app = TBRGSApp()
    app.mainloop()
    save_config(app.cfg)


if __name__ == "__main__":
    main()

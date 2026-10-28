from BMIRG_Lib import Framework
from BMIRG_Lib import _2DFM
from BMIRG_Lib import Modelling

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pathlib
import numpy as np
import json
import os
import pandas as pd
from matplotlib.transforms import Bbox


class _2DFM_Studio(Framework):
    def __init__(self, root):
        super().__init__(root, title="2DES Studio", versionNo="1.0")

        self._2DFM = _2DFM()
        self.model = Modelling()

        self.scrollLock = True

        self.em_slice_col = "k"
        self.ex_slice_col = "r"
        self.fit_col = "steelblue"
        self.resid_col = "green"
        self.focus_col = "m"
        self.em_step = 1
        self.ex_step = 1

        self.plot_settings_window = None
        self.plot_settings_vars = {}
        self.xy_mesh = None
        self.current_data_file = None

        # Peak picking / export state
        self.pick_peak_mode = False
        self.snap_to_local_max = tk.BooleanVar(value=True)
        self.selected_peaks = []
        self.selected_peak_markers = []

        local_appdata = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home()))
        self.settings_dir = local_appdata / "2DES Studio"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.settings_dir / "plot_settings.json"

        self.load_plot_settings()
        self.create_gui()

    def create_gui(self):
        super().create_gui()

        tab_definitions = (
            "view-p- 3,3,0.5,0.3 - ax1/0/1:3/True/scroll_event&self.onscroll ,ax2/1:3/0/True/scroll_event&self.onscroll , ax3/1:3/1:3/False/button_press_event&self.on_plot_click ",
            "fit-p- 2,3,0.4,0.3 - ax1/0/0:2/True/scroll_event&self.onscroll,ax2/0/2/False/None ,ax3/1/0:2/True/scroll_event&self.onscroll ,ax4/1/2/True/None "
        )

        self.build_tab_control(tab_definitions)
        self.create_bottom_bar()
        self.bind_menu_accelerator_keys()

        self.settings_button = ttk.Button(
            self.top_bar,
            text="Plot Settings",
            command=self.open_plot_settings_window
        )
        self.settings_button.pack(side="left", padx=4)

        self.pick_peak_button = ttk.Button(
            self.top_bar,
            text="Pick Peak",
            command=self.toggle_pick_peak_mode
        )
        self.pick_peak_button.pack(side="left", padx=4)

        self.snap_check = ttk.Checkbutton(
            self.top_bar,
            text="Snap to local max",
            variable=self.snap_to_local_max
        )
        self.snap_check.pack(side="left", padx=4)

        self.export_peak_button = ttk.Button(
            self.top_bar,
            text="Export Peak Data",
            command=self.export_selected_peak_data
        )
        self.export_peak_button.pack(side="left", padx=4)

        self.clear_peaks_button = ttk.Button(
            self.top_bar,
            text="Clear Peaks",
            command=self.clear_peaks
        )
        self.clear_peaks_button.pack(side="left", padx=4)

        self.export_button = ttk.Button(
            self.top_bar,
            text="Export 2D PNG",
            command=self.export_2d_png
        )
        self.export_button.pack(side="left", padx=4)

        self.button = ttk.Button(
            self.top_bar,
            text="Update Model",
            command=self.update_model
        )
        self.button.pack(side="right", padx=4)

    def onscroll(self, event):
        if self.scrollLock or not self._2DFM.is_loaded:
            return

        if event.inaxes == self.tabs["view"]["ax2"] or event.inaxes == self.tabs["fit"]["ax1"]:
            if event.button == "up":
                self._2DFM.ex_ind = (self._2DFM.ex_ind + self.em_step) % self._2DFM.data.shape[1]
            else:
                self._2DFM.ex_ind = (self._2DFM.ex_ind - self.em_step) % self._2DFM.data.shape[1]

            self.model.ex_ind = self._2DFM.ex_ind

            self._2DFM.update(self.tabs["view"], "ax2", "ax1", "ax3")
            self._2DFM.update(self.tabs["fit"], "ax1", "ax3", "ax2")
            if self.model.is_loaded:
                self.model.update(self.tabs["fit"], "ax1", "ax3", "ax4")
            self._redraw_peak_marker_if_needed()
            self.update()

        elif event.inaxes == self.tabs["view"]["ax1"] or event.inaxes == self.tabs["fit"]["ax3"]:
            if event.button == "up":
                self._2DFM.em_ind = (self._2DFM.em_ind + self.ex_step) % self._2DFM.data.shape[0]
            else:
                self._2DFM.em_ind = (self._2DFM.em_ind - self.ex_step) % self._2DFM.data.shape[0]

            self.model.em_ind = self._2DFM.em_ind

            self._2DFM.update(self.tabs["view"], "ax2", "ax1", "ax3")
            self._2DFM.update(self.tabs["fit"], "ax1", "ax3", "ax2")
            if self.model.is_loaded:
                self.model.update(self.tabs["fit"], "ax1", "ax3", "ax4")
            self._redraw_peak_marker_if_needed()
            self.update()

    def create_menu(self):
        self.menubar = tk.Menu(self.root)
        menu_definitions = (
            "File- Open Data/Ctrl+O/self.on_open_data_clicked,sep, Open Model/Ctrl+M/self.on_open_model_clicked, sep, Exit/Alt+F4/self.on_close_menu_clicked",
            "About- About/F1/self.on_about_menu_clicked"
        )
        self.build_menu(menu_definitions)

    def bind_menu_accelerator_keys(self):
        self.root.bind("<KeyPress-F1>", self.on_about_menu_clicked)
        self.root.bind("<Control-M>", self.on_open_model_clicked)
        self.root.bind("<Control-m>", self.on_open_model_clicked)
        self.root.bind("<Control-O>", self.on_open_data_clicked)
        self.root.bind("<Control-o>", self.on_open_data_clicked)

    def on_open_data_clicked(self, event=None):
        p = filedialog.askopenfilename()
        if p == "":
            return

        if self._2DFM.is_loaded:
            self._clear_experimental_plots()

        if self.model.is_loaded:
            self._clear_model_plots()
            self.model.is_loaded = False
            self.model.data = None
            self.model.eval_res = None
            self.model.adj_r2 = None
            self.xy_mesh = None

        self._clear_selected_peak()

        self._2DFM.em_ind = 0
        self._2DFM.ex_ind = 0
        self.model.em_ind = 0
        self.model.ex_ind = 0

        path = pathlib.Path(p)
        self.current_data_file = path
        self._2DFM.open(path)

        self._draw_experimental_plots()

        self.tabs["view"]["canvas"].draw()
        self.tabs["fit"]["canvas"].draw()
        self.scrollLock = False

    def on_open_model_clicked(self, event=None):
        p = filedialog.askopenfilename()
        if p == "":
            return

        if not self._2DFM.is_loaded:
            messagebox.showwarning("No data loaded", "Please load experimental data before loading a model.")
            return

        try:
            if self.model.is_loaded:
                self._clear_model_plots()

            path = pathlib.Path(p)
            self.model.open(path, None)

            self.model.set_ex_em_range(self._2DFM.ex_wavelengths, self._2DFM.em_wavelengths)
            self.model.build_mask_from_data(offset_nm=40)

            self.xy_mesh = np.meshgrid(self._2DFM.ex_wavelengths, self._2DFM.em_wavelengths)

            self.model.generateModel()
            self.model.generate_Params(data_for_scale=self._2DFM.data.to_numpy(dtype=float))
            self.model.eval(self._2DFM.data.to_numpy(dtype=float), self.xy_mesh)

            self.model.em_ind = self._2DFM.em_ind
            self.model.ex_ind = self._2DFM.ex_ind

            self._draw_model_plots()
            self.tabs["fit"]["canvas"].draw()

        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))

    def update_model(self, event=None):
        if self.model.is_loaded and self.xy_mesh is not None:
            try:
                self.model.refresh(self._2DFM.data.to_numpy(dtype=float), self.xy_mesh, self.tabs["fit"], "ax1", "ax3", "ax4")
                self.tabs["fit"]["canvas"].draw()
                self.update()
            except Exception as exc:
                messagebox.showerror("Update Model Error", str(exc))

    def toggle_pick_peak_mode(self):
        if not self._2DFM.is_loaded:
            messagebox.showwarning("No data loaded", "Please load a data file before selecting a peak.")
            return

        self.pick_peak_mode = not self.pick_peak_mode
        self.pick_peak_button.configure(text="Picking..." if self.pick_peak_mode else "Pick Peak")

    def on_plot_click(self, event):
        if not self.pick_peak_mode:
            return
        if not self._2DFM.is_loaded:
            return
        if event.inaxes != self.tabs["view"]["ax3"]:
            return
        if event.xdata is None or event.ydata is None:
            return

        clicked_ex_idx = self._2DFM.nearest_ex_index(event.xdata)
        clicked_em_idx = self._2DFM.nearest_em_index(event.ydata)

        clicked_ex = float(self._2DFM.ex_wavelengths[clicked_ex_idx])
        clicked_em = float(self._2DFM.em_wavelengths[clicked_em_idx])
        clicked_intensity = self._2DFM.get_intensity(clicked_em_idx, clicked_ex_idx)

        if self.snap_to_local_max.get():
            local = self._2DFM.find_local_max(clicked_em_idx, clicked_ex_idx, em_half_window=2, ex_half_window=1)
            sel_em_idx = local["em_index"]
            sel_ex_idx = local["ex_index"]
        else:
            local = None
            sel_em_idx = clicked_em_idx
            sel_ex_idx = clicked_ex_idx

        sel_ex = float(self._2DFM.ex_wavelengths[sel_ex_idx])
        sel_em = float(self._2DFM.em_wavelengths[sel_em_idx])
        sel_intensity = self._2DFM.get_intensity(sel_em_idx, sel_ex_idx)

        # Move slice indicators to selected point
        self._2DFM.ex_ind = sel_ex_idx
        self._2DFM.em_ind = sel_em_idx
        self.model.ex_ind = sel_ex_idx
        self.model.em_ind = sel_em_idx

        self._2DFM.update(self.tabs["view"], "ax2", "ax1", "ax3")
        self._2DFM.update(self.tabs["fit"], "ax1", "ax3", "ax2")
        if self.model.is_loaded:
            self.model.update(self.tabs["fit"], "ax1", "ax3", "ax4")

        ex_slice_w, ex_slice_i = self._2DFM.get_excitation_slice(sel_em_idx)
        em_slice_w, em_slice_i = self._2DFM.get_emission_slice(sel_ex_idx)

        ex_norm = self._2DFM.normalize_slice(ex_slice_i)
        em_norm = self._2DFM.normalize_slice(em_slice_i)

        e_ex = self._2DFM.nm_to_ev(sel_ex)
        e_em = self._2DFM.nm_to_ev(sel_em)

        peak_number = len(self.selected_peaks) + 1

        peak_data = {
            "peak_id": peak_number,
            "file_name": self.current_data_file.name if self.current_data_file is not None else "",
            "file_stem": self.current_data_file.stem if self.current_data_file is not None else "peak_export",
            "clicked_lambda_ex_nm": clicked_ex,
            "clicked_lambda_em_nm": clicked_em,
            "clicked_intensity": clicked_intensity,
            "selected_lambda_ex_nm": sel_ex,
            "selected_lambda_em_nm": sel_em,
            "selected_intensity": sel_intensity,
            "selected_ex_index": sel_ex_idx,
            "selected_em_index": sel_em_idx,
            "excitation_photon_energy_eV": e_ex,
            "emission_photon_energy_eV": e_em,
            "stokes_shift_nm": sel_em - sel_ex,
            "stokes_shift_eV": e_ex - e_em,
            "ex_slice_wavelengths_nm": ex_slice_w,
            "ex_slice_intensity": ex_slice_i,
            "ex_slice_intensity_normalized": ex_norm,
            "em_slice_wavelengths_nm": em_slice_w,
            "em_slice_intensity": em_slice_i,
            "em_slice_intensity_normalized": em_norm,
            "local_window_df": self._2DFM.get_local_window_dataframe(sel_em_idx, sel_ex_idx, em_half_window=2, ex_half_window=1),
            "local_max_used": self.snap_to_local_max.get(),
        }

        self.selected_peaks.append(peak_data)

        self._draw_selected_peak_marker(sel_ex, sel_em)

        self.pick_peak_mode = False
        self.pick_peak_button.configure(text="Pick Peak")

        self.tabs["view"]["canvas"].draw()
        self.tabs["fit"]["canvas"].draw()

        messagebox.showinfo(
            "Peak Selected",
            f"Peak {peak_number} selected at:\n"
            f"Ex = {sel_ex:.1f} nm\n"
            f"Em = {sel_em:.1f} nm\n"
            f"Intensity = {sel_intensity:.2f}"
        )

    def _draw_selected_peak_marker(self, ex_wav, em_wav):
        ax = self.tabs["view"]["ax3"]

        marker = ax.plot(
            [ex_wav],
            [em_wav],
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=2,
            linestyle="None",
            zorder=10
        )[0]

        self.selected_peak_markers.append(marker)

    def _redraw_peak_marker_if_needed(self):
        if not self.selected_peaks:
            return

        for marker in self.selected_peak_markers:
            try:
                marker.remove()
            except Exception:
                pass
        self.selected_peak_markers = []

        for peak in self.selected_peaks:
            self._draw_selected_peak_marker(
                peak["selected_lambda_ex_nm"],
                peak["selected_lambda_em_nm"]
            )

        self.tabs["view"]["canvas"].draw_idle()

    def _clear_selected_peak(self):
        self.selected_peaks = []

        for marker in self.selected_peak_markers:
            try:
                marker.remove()
            except Exception:
                pass

        self.selected_peak_markers = []

    def clear_peaks(self):
        self._clear_selected_peak()
        self.tabs["view"]["canvas"].draw()

    def export_selected_peak_data(self):
        if not self._2DFM.is_loaded:
            messagebox.showwarning("No data loaded", "Please load a data file first.")
            return

        if not self.selected_peaks:
            messagebox.showwarning("No peaks selected", "Please use 'Pick Peak' to select one or more peaks first.")
            return

        base_stem = self.selected_peaks[0]["file_stem"]
        initial_dir = str(self.current_data_file.parent) if self.current_data_file is not None else str(pathlib.Path.home())
        default_name = f"{base_stem}_peak_data.xlsx"

        save_path = filedialog.asksaveasfilename(
            title="Export Peak Data",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            initialfile=default_name,
            initialdir=initial_dir,
        )

        if save_path == "":
            return

        try:
            summary_rows = []
            for peak in self.selected_peaks:
                summary_rows.append({
                    "source_file": peak["file_name"],
                    "selected_lambda_ex_nm": peak["selected_lambda_ex_nm"],
                    "selected_lambda_em_nm": peak["selected_lambda_em_nm"],
                    "selected_intensity": peak["selected_intensity"],
                    "local_max_used": peak["local_max_used"],
                    "excitation_photon_energy_eV": peak["excitation_photon_energy_eV"],
                    "emission_photon_energy_eV": peak["emission_photon_energy_eV"],
                    "stokes_shift_nm": peak["stokes_shift_nm"],
                    "stokes_shift_eV": peak["stokes_shift_eV"],
                })

            summary_df = pd.DataFrame(summary_rows)

            with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                for peak in self.selected_peaks:
                    peak_id = peak["peak_id"]

                    ex_slice_df = pd.DataFrame({
                        "excitation_wavelength_nm": peak["ex_slice_wavelengths_nm"],
                        "intensity": peak["ex_slice_intensity"],
                        "normalized_intensity": peak["ex_slice_intensity_normalized"],
                    })

                    em_slice_df = pd.DataFrame({
                        "emission_wavelength_nm": peak["em_slice_wavelengths_nm"],
                        "intensity": peak["em_slice_intensity"],
                        "normalized_intensity": peak["em_slice_intensity_normalized"],
                    })

                    ex_sheet = f"ExSlice_P{peak_id}"
                    em_sheet = f"EmSlice_P{peak_id}"
                    local_sheet = f"Local_P{peak_id}"

                    ex_slice_df.to_excel(writer, sheet_name=ex_sheet, index=False)
                    em_slice_df.to_excel(writer, sheet_name=em_sheet, index=False)
                    peak["local_window_df"].to_excel(writer, sheet_name=local_sheet)

            messagebox.showinfo(
                "Export Peak Data",
                f"Exported {len(self.selected_peaks)} peak(s) to:\n{save_path}"
            )

        except Exception as exc:
            messagebox.showerror("Export Peak Data", f"Could not export peak data:\n{exc}")

    def export_2d_png(self):
        if not self._2DFM.is_loaded:
            messagebox.showwarning("No data loaded", "Please load a data file before exporting a PNG.")
            return

        if self.current_data_file is not None:
            default_name = f"{self.current_data_file.stem}.png"
            initial_dir = str(self.current_data_file.parent)
        else:
            default_name = "2d_heatmap.png"
            initial_dir = str(pathlib.Path.home())

        save_path = filedialog.asksaveasfilename(
            title="Export 2D Heatmap PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=default_name,
            initialdir=initial_dir,
        )

        if save_path == "":
            return

        try:
            fig = self.tabs["view"]["fig"]
            heatmap_ax = self.tabs["view"]["ax3"]
            colorbar_ax = self.tabs["view"].get("ax3_cax", None)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            bbox = heatmap_ax.get_tightbbox(renderer)

            if colorbar_ax is not None:
                cbbox = colorbar_ax.get_tightbbox(renderer)
                bbox = Bbox.union([bbox, cbbox])

            bbox = bbox.expanded(1.02, 1.02)
            bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())

            fig.savefig(save_path, dpi=300, bbox_inches=bbox_inches)
            messagebox.showinfo("Export PNG", f"Saved 2D heatmap PNG to:\n{save_path}")

        except Exception as exc:
            messagebox.showerror("Export PNG", f"Could not save PNG:\n{exc}")

    def open_settings_file(self):
        try:
            if self.settings_file.exists():
                os.startfile(str(self.settings_file))
            else:
                messagebox.showinfo(
                    "Settings File",
                    f"The settings file does not exist yet.\n\nExpected location:\n{self.settings_file}"
                )
        except Exception as exc:
            try:
                os.startfile(str(self.settings_dir))
            except Exception:
                messagebox.showerror("Open Settings File", f"Could not open settings file or folder:\n{exc}")

    def copy_settings_path(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(str(self.settings_file))
            self.root.update()
            messagebox.showinfo("Copy Settings Path", "Settings file path copied to clipboard.")
        except Exception as exc:
            messagebox.showerror("Copy Settings Path", f"Could not copy settings path:\n{exc}")

    def open_plot_settings_window(self):
        if self.plot_settings_window is not None and self.plot_settings_window.winfo_exists():
            self.plot_settings_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Plot Settings")
        win.minsize(560, 520)
        self.plot_settings_window = win

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="Plot Settings",
            font=("Segoe UI", 11, "bold")
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        subtitle = ttk.Label(
            outer,
            text="Adjust 1D slice ranges, 2D plot ranges, intensity range, and contour levels."
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(0, 10))

        container = ttk.Frame(outer)
        container.grid(row=2, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        left_col = ttk.Frame(container)
        right_col = ttk.Frame(container)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        defaults = self._current_plot_settings_defaults()
        self.plot_settings_vars = {}

        def add_range_row(parent, row, label_text, min_key, max_key):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            min_var = tk.StringVar(value=defaults.get(min_key, ""))
            max_var = tk.StringVar(value=defaults.get(max_key, ""))
            self.plot_settings_vars[min_key] = min_var
            self.plot_settings_vars[max_key] = max_var

            ttk.Entry(parent, textvariable=min_var, width=10).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
            ttk.Label(parent, text="to").grid(row=row, column=2, padx=2)
            ttk.Entry(parent, textvariable=max_var, width=10).grid(row=row, column=3, sticky="ew", padx=4, pady=3)

        emission_frame = ttk.LabelFrame(left_col, text="Emission 1D Slice")
        emission_frame.pack(fill="x", expand=False, pady=(0, 10))
        add_range_row(emission_frame, 0, "X range", "em_xmin", "em_xmax")
        add_range_row(emission_frame, 1, "Y range", "em_ymin", "em_ymax")
        for col in range(4):
            emission_frame.columnconfigure(col, weight=1)

        excitation_frame = ttk.LabelFrame(left_col, text="Excitation 1D Slice")
        excitation_frame.pack(fill="x", expand=False, pady=(0, 10))
        add_range_row(excitation_frame, 0, "X range", "ex_xmin", "ex_xmax")
        add_range_row(excitation_frame, 1, "Y range", "ex_ymin", "ex_ymax")
        for col in range(4):
            excitation_frame.columnconfigure(col, weight=1)

        plot2d_frame = ttk.LabelFrame(right_col, text="2D Plot Axes")
        plot2d_frame.pack(fill="x", expand=False, pady=(0, 10))
        add_range_row(plot2d_frame, 0, "X range", "plot2d_xmin", "plot2d_xmax")
        add_range_row(plot2d_frame, 1, "Y range", "plot2d_ymin", "plot2d_ymax")
        for col in range(4):
            plot2d_frame.columnconfigure(col, weight=1)

        colour_frame = ttk.LabelFrame(right_col, text="2D Colour / Contours")
        colour_frame.pack(fill="x", expand=False, pady=(0, 10))

        ttk.Label(colour_frame, text="Intensity range").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        int_min_var = tk.StringVar(value=defaults.get("int_min", ""))
        int_max_var = tk.StringVar(value=defaults.get("int_max", ""))
        self.plot_settings_vars["int_min"] = int_min_var
        self.plot_settings_vars["int_max"] = int_max_var
        ttk.Entry(colour_frame, textvariable=int_min_var, width=10).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(colour_frame, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(colour_frame, textvariable=int_max_var, width=10).grid(row=0, column=3, sticky="ew", padx=4, pady=3)

        ttk.Label(colour_frame, text="Contour levels").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        contour_var = tk.StringVar(value=defaults.get("contour_levels", "50"))
        self.plot_settings_vars["contour_levels"] = contour_var
        ttk.Entry(colour_frame, textvariable=contour_var, width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=3)

        for col in range(4):
            colour_frame.columnconfigure(col, weight=1)

        hint = ttk.Label(
            outer,
            text="Tip: leave both boxes blank for any range you want to auto-scale.",
            foreground="#555555"
        )
        hint.grid(row=3, column=0, sticky="w", pady=(6, 10))

        settings_path_label = ttk.Label(
            outer,
            text=f"Settings file: {self.settings_file}",
            foreground="#555555",
            wraplength=520
        )
        settings_path_label.grid(row=4, column=0, sticky="w", pady=(0, 10))

        button_frame = ttk.Frame(outer)
        button_frame.grid(row=5, column=0, sticky="e")

        ttk.Button(button_frame, text="Open Settings File", command=self.open_settings_file).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Copy Settings Path", command=self.copy_settings_path).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Apply", command=self.apply_plot_settings).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Close", command=win.destroy).pack(side="left", padx=4)

    def _current_plot_settings_defaults(self):
        def unpack_range(rng, index):
            if rng is None:
                return ""
            return str(rng[index])

        return {
            "em_xmin": unpack_range(self._2DFM.em_xrange, 0),
            "em_xmax": unpack_range(self._2DFM.em_xrange, 1),
            "em_ymin": unpack_range(self._2DFM.em_yrange, 0),
            "em_ymax": unpack_range(self._2DFM.em_yrange, 1),
            "ex_xmin": unpack_range(self._2DFM.ex_xrange, 0),
            "ex_xmax": unpack_range(self._2DFM.ex_xrange, 1),
            "ex_ymin": unpack_range(self._2DFM.ex_yrange, 0),
            "ex_ymax": unpack_range(self._2DFM.ex_yrange, 1),
            "plot2d_xmin": unpack_range(self._2DFM.plot2d_xrange, 0),
            "plot2d_xmax": unpack_range(self._2DFM.plot2d_xrange, 1),
            "plot2d_ymin": unpack_range(self._2DFM.plot2d_yrange, 0),
            "plot2d_ymax": unpack_range(self._2DFM.plot2d_yrange, 1),
            "int_min": unpack_range(self._2DFM.intensity_range, 0),
            "int_max": unpack_range(self._2DFM.intensity_range, 1),
            "contour_levels": str(self._2DFM.contour_levels),
        }

    def _parse_optional_float(self, value):
        value = value.strip()
        if value == "":
            return None
        return float(value)

    def _parse_range(self, min_key, max_key):
        vmin = self._parse_optional_float(self.plot_settings_vars[min_key].get())
        vmax = self._parse_optional_float(self.plot_settings_vars[max_key].get())

        if vmin is None and vmax is None:
            return None
        if vmin is None or vmax is None:
            raise ValueError(f"Both {min_key} and {max_key} must be provided, or both left blank.")

        return (vmin, vmax)

    def apply_plot_settings(self):
        try:
            em_xrange = self._parse_range("em_xmin", "em_xmax")
            em_yrange = self._parse_range("em_ymin", "em_ymax")
            ex_xrange = self._parse_range("ex_xmin", "ex_xmax")
            ex_yrange = self._parse_range("ex_ymin", "ex_ymax")
            plot2d_xrange = self._parse_range("plot2d_xmin", "plot2d_xmax")
            plot2d_yrange = self._parse_range("plot2d_ymin", "plot2d_ymax")
            intensity_range = self._parse_range("int_min", "int_max")

            contour_text = self.plot_settings_vars["contour_levels"].get().strip()
            contour_levels = int(contour_text) if contour_text else 50
            if contour_levels < 2:
                raise ValueError("Contour levels must be at least 2.")

            self._2DFM.em_xrange = em_xrange
            self._2DFM.em_yrange = em_yrange
            self._2DFM.ex_xrange = ex_xrange
            self._2DFM.ex_yrange = ex_yrange
            self._2DFM.plot2d_xrange = plot2d_xrange
            self._2DFM.plot2d_yrange = plot2d_yrange
            self._2DFM.intensity_range = intensity_range
            self._2DFM.contour_levels = contour_levels

            self.model.em_xrange = em_xrange
            self.model.em_yrange = em_yrange
            self.model.ex_xrange = ex_xrange
            self.model.ex_yrange = ex_yrange
            self.model.plot2d_xrange = plot2d_xrange
            self.model.plot2d_yrange = plot2d_yrange
            self.model.intensity_range = intensity_range
            self.model.contour_levels = contour_levels

            self.save_plot_settings()
            self.redraw_all_plots()
            self.update()

        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))

    def save_plot_settings(self):
        settings = {
            "em_xrange": self._2DFM.em_xrange,
            "em_yrange": self._2DFM.em_yrange,
            "ex_xrange": self._2DFM.ex_xrange,
            "ex_yrange": self._2DFM.ex_yrange,
            "plot2d_xrange": self._2DFM.plot2d_xrange,
            "plot2d_yrange": self._2DFM.plot2d_yrange,
            "intensity_range": self._2DFM.intensity_range,
            "contour_levels": self._2DFM.contour_levels,
        }

        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as exc:
            messagebox.showwarning("Settings Warning", f"Could not save plot settings:\n{exc}")

    def load_plot_settings(self):
        if not self.settings_file.exists():
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)

            self._2DFM.em_xrange = tuple(settings["em_xrange"]) if settings.get("em_xrange") is not None else None
            self._2DFM.em_yrange = tuple(settings["em_yrange"]) if settings.get("em_yrange") is not None else None
            self._2DFM.ex_xrange = tuple(settings["ex_xrange"]) if settings.get("ex_xrange") is not None else None
            self._2DFM.ex_yrange = tuple(settings["ex_yrange"]) if settings.get("ex_yrange") is not None else None
            self._2DFM.plot2d_xrange = tuple(settings["plot2d_xrange"]) if settings.get("plot2d_xrange") is not None else None
            self._2DFM.plot2d_yrange = tuple(settings["plot2d_yrange"]) if settings.get("plot2d_yrange") is not None else None
            self._2DFM.intensity_range = tuple(settings["intensity_range"]) if settings.get("intensity_range") is not None else None
            self._2DFM.contour_levels = settings.get("contour_levels", 50)

        except Exception as exc:
            messagebox.showwarning("Settings Warning", f"Could not load plot settings:\n{exc}")

    def redraw_all_plots(self):
        if self._2DFM.is_loaded:
            self._clear_experimental_plots()
            self._draw_experimental_plots()

        if self.model.is_loaded and self.model.eval_res is not None:
            self._clear_model_plots()
            self._draw_model_plots()

        self._redraw_peak_marker_if_needed()

        self.tabs["view"]["canvas"].draw()
        self.tabs["fit"]["canvas"].draw()

    def _draw_experimental_plots(self):
        self._2DFM.plot(self.tabs["view"], "ax3", axisLabels=True)
        self._2DFM.indicators(self.tabs["view"], "ax3", animated=True)
        self._2DFM.explot(self.tabs["view"], "ax1")
        self._2DFM.emplot(self.tabs["view"], "ax2", rotate=True, invert_y_ax=True)

        self._2DFM.plot(self.tabs["fit"], "ax2", title="Experimental Data")
        self._2DFM.emplot(
            self.tabs["fit"],
            "ax1",
            y_label=f"Intensity (Ex. {self._2DFM.ex_wavelengths[self._2DFM.ex_ind]} nm)"
        )
        self._2DFM.explot(
            self.tabs["fit"],
            "ax3",
            x_label="Wavelength (nm)",
            y_label=f"Intensity (Em. {self._2DFM.em_wavelengths[self._2DFM.em_ind]} nm)"
        )
        self._2DFM.indicators(self.tabs["fit"], "ax2", animated=True)

    def _draw_model_plots(self):
        self.model.plot(self.tabs["fit"], "ax4", title=f"Model Adj-R squared: {self.model.adj_r2:.2f}")
        self.model.emplot(self.tabs["fit"], "ax1")
        self.model.explot(self.tabs["fit"], "ax3")
        self.model.indicators(self.tabs["fit"], "ax4", animated=True)

    def _clear_experimental_plots(self):
        self._safe_remove_colorbar(self.tabs["view"], "ax3")
        self._safe_remove_indicator_artists(self.tabs["view"], "ax3")
        self.tabs["view"]["ax1"].clear()
        self.tabs["view"]["ax2"].clear()
        self.tabs["view"]["ax3"].clear()

        self._safe_remove_colorbar(self.tabs["fit"], "ax2")
        self._safe_remove_indicator_artists(self.tabs["fit"], "ax2")
        self.tabs["fit"]["ax1"].clear()
        self.tabs["fit"]["ax2"].clear()
        self.tabs["fit"]["ax3"].clear()

    def _clear_model_plots(self):
        self._safe_remove_colorbar(self.tabs["fit"], "ax4")
        self._safe_remove_indicator_artists(self.tabs["fit"], "ax4")
        self.tabs["fit"]["ax4"].clear()

        self._safe_remove_line(self.tabs["fit"], "ax1_em_model")
        self._safe_remove_line(self.tabs["fit"], "ax3_ex_model")

    def _safe_remove_colorbar(self, tab_dict, ax_name):
        cb_key = f"{ax_name}_cb"
        if cb_key in tab_dict:
            try:
                tab_dict[cb_key].remove()
            except Exception:
                pass
            finally:
                tab_dict.pop(cb_key, None)

    def _safe_remove_indicator_artists(self, tab_dict, ax_name):
        for suffix in ("em_ind", "ex_ind"):
            key = f"{ax_name}_{suffix}"
            if key in tab_dict:
                try:
                    tab_dict["blit_manager"].remove_artist(tab_dict[key])
                except Exception:
                    pass
                finally:
                    tab_dict.pop(key, None)

    def _safe_remove_line(self, tab_dict, key):
        if key in tab_dict:
            try:
                line_list = tab_dict[key]
                if line_list and len(line_list) > 0:
                    line_list[0].remove()
            except Exception:
                pass
            finally:
                tab_dict.pop(key, None)

    def update_plot(self):
        pass

    def on_close_menu_clicked(self):
        self.root.destroy()

    def on_about_menu_clicked(self, event=None):
        self.create_about_window(
            "Daniel J. Oliver & Carole C. Perry",
            "Interdisciplinary Biomedical Research Centre, "
            "School of Science and Technology, Nottingham Trent University, "
            "Clifton Lane, Nottingham NG11 8NS, United Kingdom."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = _2DFM_Studio(root)
    root.mainloop()
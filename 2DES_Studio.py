from BMIRG_Lib import Framework
from BMIRG_Lib import _2DFM
from BMIRG_Lib import Modelling

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pathlib
import numpy as np
import json
import os


class _2DFM_Studio(Framework):
    def __init__(self, root):
        super().__init__(root, title="2DES Studio", versionNo="1.0")
        self.create_gui()

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

        local_appdata = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home()))
        self.settings_dir = local_appdata / "2DES Studio"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.settings_dir / "plot_settings.json"

        self.load_plot_settings()

    def create_gui(self):
        super().create_gui()

        tab_definitions = (
            "view-p- 3,3,0.5,0.3 - ax1/0/1:3/True/scroll_event&self.onscroll ,ax2/1:3/0/True/scroll_event&self.onscroll , ax3/1:3/1:3/False/None ",
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

        self._2DFM.em_ind = 0
        self._2DFM.ex_ind = 0
        self.model.em_ind = 0
        self.model.ex_ind = 0

        path = pathlib.Path(p)
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

        if self.model.is_loaded:
            self._clear_model_plots()

        path = pathlib.Path(p)
        self.model.open(path, "M0.csv")

        self.model.set_ex_em_range(self._2DFM.ex_wavelengths, self._2DFM.em_wavelengths)
        self.xy_mesh = np.meshgrid(self._2DFM.ex_wavelengths, self._2DFM.em_wavelengths)
        self.model.generateModel()
        self.model.generate_Params()
        self.model.eval(self._2DFM.data, self.xy_mesh)

        self._draw_model_plots()
        self.tabs["fit"]["canvas"].draw()

    def update_model(self, event=None):
        if self.model.is_loaded:
            self.model.refresh(self._2DFM.data, self.xy_mesh, self.tabs["fit"], "ax1", "ax3", "ax4")
            self.tabs["fit"]["canvas"].draw()
            self.update()

    def open_settings_folder(self):
        try:
            if self.settings_file.exists():
                os.startfile(str(self.settings_file.parent))
            else:
                os.startfile(str(self.settings_dir))
        except Exception as exc:
            messagebox.showerror("Open Settings Folder", f"Could not open settings folder:\n{exc}")

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

        ttk.Button(button_frame, text="Open Settings Folder", command=self.open_settings_folder).pack(side="left", padx=4)
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

            self.model.em_xrange = self._2DFM.em_xrange
            self.model.em_yrange = self._2DFM.em_yrange
            self.model.ex_xrange = self._2DFM.ex_xrange
            self.model.ex_yrange = self._2DFM.ex_yrange
            self.model.plot2d_xrange = self._2DFM.plot2d_xrange
            self.model.plot2d_yrange = self._2DFM.plot2d_yrange
            self.model.intensity_range = self._2DFM.intensity_range
            self.model.contour_levels = self._2DFM.contour_levels

        except Exception as exc:
            messagebox.showwarning("Settings Warning", f"Could not load plot settings:\n{exc}")

    def redraw_all_plots(self):
        if self._2DFM.is_loaded:
            self._clear_experimental_plots()
            self._draw_experimental_plots()

        if self.model.is_loaded:
            self._clear_model_plots()
            self._draw_model_plots()

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
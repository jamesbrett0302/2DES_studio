# Modified to add configurable 1D/2D plot ranges, intensity limits, and contour levels
# for 2DES Studio. Modernised for improved compatibility and robustness.

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from scipy.constants import speed_of_light


class Spectrum:
    def __init__(self, data):
        self.data = data
        self.is_loaded = data is not None

        self.SIprefix = {
            "yotta": ["Y", 1E24], "zetta": ["Z", 1E21], "exa": ["E", 1E18],
            "peta": ["P", 1E15], "tera": ["T", 1E12], "giga": ["G", 1E9],
            "mega": ["M", 1E6], "kilo": ["k", 1E3], "hecto": ["h", 1E2],
            "deka": ["da", 1E1], "deci": ["d", 1E-1], "centi": ["c", 1E-2],
            "milli": ["m", 1E-3], "micro": ["", 1E-6], "nano": ["n", 1E-9],
            "pico": ["p", 1E-12], "femto": ["f", 1E-15], "atto": ["a", 1E-18],
            "zepto": ["z", 1E-21], "yocto": ["y", 1E-24]
        }

    def nm2Hz(self, wavelength, input_si_prefix="nano"):
        wavel = wavelength * self.SIprefix[input_si_prefix][1]
        return speed_of_light / wavel

    def Hz2nm(self, frequency, output_si_prefix="nano"):
        wavel = speed_of_light / frequency
        return wavel / self.SIprefix[output_si_prefix][1]


class _2DFM(Spectrum):
    def __init__(self):
        super().__init__(None)
        self.ex_wavelengths = None
        self.em_wavelengths = None

        self.em_ind = 0
        self.ex_ind = 0

        self.aa = True
        self.alpha = 1
        self.facecolour = (0, 0, 1)
        self.cmap = "viridis"

        self.em_xrange = None
        self.em_yrange = None
        self.ex_xrange = None
        self.ex_yrange = None
        self.plot2d_xrange = None
        self.plot2d_yrange = None
        self.intensity_range = None
        self.contour_levels = 50

    def open(self, file):
        try:
            self.data = pd.read_csv(file, header=None)
        except Exception as exc:
            raise Exception(f"Dataset could not be loaded: {exc}")

        self.is_loaded = True

        self.ex_wavelengths = self.data.loc[0].values.tolist()
        del self.ex_wavelengths[0]
        self.data.drop(index=0, inplace=True)

        self.em_wavelengths = self.data.iloc[:, 0].values.tolist()
        self.data.drop(columns=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        self.data = self.data.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.data[self.data < 0] = 0

    def clear(self, tabDict, emPlot_ax, exPlot_ax, cont_ax):
        if f"{cont_ax}_cb" in tabDict:
            tabDict[f"{cont_ax}_cb"].remove()
        if f"{cont_ax}_em_ind" in tabDict:
            tabDict["blit_manager"].remove_artist(tabDict[f"{cont_ax}_em_ind"])
        if f"{cont_ax}_ex_ind" in tabDict:
            tabDict["blit_manager"].remove_artist(tabDict[f"{cont_ax}_ex_ind"])

        tabDict[cont_ax].clear()
        tabDict[emPlot_ax].clear()
        tabDict[exPlot_ax].clear()

    def _build_contour_levels(self, zdata):
        n_levels = max(2, int(self.contour_levels))

        if self.intensity_range is not None:
            vmin, vmax = self.intensity_range
        else:
            vmin = float(np.nanmin(zdata))
            vmax = float(np.nanmax(zdata))

        if vmax <= vmin:
            vmax = vmin + 1.0

        levels = np.linspace(vmin, vmax, n_levels)
        return levels, vmin, vmax

    def plot(self, tabDict, ax, axisLabels=False, title=None):
        zdata = self.data.astype("float64").values
        levels, vmin, vmax = self._build_contour_levels(zdata)

        psm = tabDict[ax].contourf(
            self.ex_wavelengths,
            self.em_wavelengths,
            zdata,
            levels=levels,
            antialiased=self.aa,
            alpha=self.alpha,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if self.plot2d_xrange is not None:
            tabDict[ax].set_xlim(*self.plot2d_xrange)
        else:
            tabDict[ax].set_xlim(230, 770)

        if self.plot2d_yrange is not None:
            ymin, ymax = self.plot2d_yrange
            tabDict[ax].set_ylim(ymax, ymin)
        else:
            tabDict[ax].set_ylim(800, 280)

        tabDict[ax].set_xticks(np.arange(250, 771, 100))
        tabDict[ax].set_yticks(np.arange(300, 901, 100))
        tabDict[ax].set_facecolor(self.facecolour)

        if axisLabels:
            tabDict[ax].set_xlabel("Excitation Wavelength (nm)")
            tabDict[ax].set_ylabel("Emission Wavelength (nm)")

        tabDict[f"{ax}_orig_loc"] = tabDict[ax].get_axes_locator()
        tabDict[f"{ax}_cax"] = make_axes_locatable(tabDict[ax]).append_axes("right", size="5%", pad=0.05)
        tabDict[f"{ax}_cb"] = tabDict["fig"].colorbar(psm, cax=tabDict[f"{ax}_cax"])
        tabDict[f"{ax}_cb"].set_label("Intensity")

        if title is not None:
            tabDict[ax].set_title(title)

    def emplot(self, tabDict, ax, colour="k", rotate=False, invert_x_ax=False, invert_y_ax=False, x_label=None, y_label=None):
        if rotate:
            tabDict[f"{ax}_rotate"] = True
            tabDict[f"{ax}_em_data"] = tabDict[ax].plot(
                self.data.iloc[:, self.ex_ind],
                self.em_wavelengths,
                color=colour,
                linewidth=1
            )
            tabDict[ax].invert_xaxis()
        else:
            tabDict[f"{ax}_rotate"] = False
            tabDict[f"{ax}_em_data"] = tabDict[ax].plot(
                self.em_wavelengths,
                self.data.iloc[:, self.ex_ind],
                color=colour,
                linewidth=1
            )

        tabDict[ax].set_title("Emission 1D-Spectral Slice")
        tabDict[f"{ax}_em_data_ylabel"] = y_label

        if invert_x_ax:
            tabDict[ax].invert_xaxis()
        if invert_y_ax:
            tabDict[ax].invert_yaxis()
        if x_label is not None:
            tabDict[ax].set_xlabel(x_label)
        if y_label is not None:
            tabDict[ax].set_ylabel(y_label)

        self.apply_em_plot_limits(tabDict, ax)

    def explot(self, tabDict, ax, colour="r", x_label=None, y_label=None):
        tabDict[f"{ax}_ex_data"] = tabDict[ax].plot(
            self.ex_wavelengths,
            self.data.iloc[self.em_ind, :],
            color=colour,
            linewidth=1
        )
        tabDict[f"{ax}_ex_data_ylabel"] = y_label
        tabDict[ax].set_xticks(range(250, 771, 100))
        tabDict[ax].set_title("Excitation 1D-Spectral Slice")

        if x_label is not None:
            tabDict[ax].set_xlabel(x_label)
        if y_label is not None:
            tabDict[ax].set_ylabel(y_label)

        self.apply_ex_plot_limits(tabDict, ax)

    def indicators(self, tabDict, ax, em_colour="k", ex_colour="r", animated=False):
        tabDict[f"{ax}_em_ind"] = tabDict[ax].axvline(
            x=self.ex_wavelengths[self.ex_ind],
            color=em_colour,
            linewidth=2,
            linestyle="--",
            animated=animated,
            visible=True
        )
        tabDict[f"{ax}_ex_ind"] = tabDict[ax].axhline(
            y=self.em_wavelengths[self.em_ind],
            color=ex_colour,
            linewidth=2,
            linestyle="--",
            animated=animated,
            visible=True
        )
        if animated:
            tabDict["blit_manager"].add_artist(tabDict[f"{ax}_em_ind"])
            tabDict["blit_manager"].add_artist(tabDict[f"{ax}_ex_ind"])

    def apply_em_plot_limits(self, tabDict, ax):
        if self.em_xrange is not None:
            xmin, xmax = self.em_xrange
            if tabDict.get(f"{ax}_rotate", False):
                tabDict[ax].set_xlim(xmax, xmin)
            else:
                tabDict[ax].set_xlim(xmin, xmax)

        if self.em_yrange is not None:
            tabDict[ax].set_ylim(*self.em_yrange)

    def apply_ex_plot_limits(self, tabDict, ax):
        if self.ex_xrange is not None:
            tabDict[ax].set_xlim(*self.ex_xrange)
        if self.ex_yrange is not None:
            tabDict[ax].set_ylim(*self.ex_yrange)

    def update(self, tabDict, emPlot_ax, exPlot_ax, cont_ax):
        if self.data is None:
            return

        tabDict[f"{cont_ax}_em_ind"].set_xdata([self.ex_wavelengths[self.ex_ind]])
        tabDict[f"{cont_ax}_ex_ind"].set_ydata([self.em_wavelengths[self.em_ind]])

        tabDict[f"{exPlot_ax}_ex_data"][0].set_ydata(self.data.iloc[self.em_ind, :])

        if tabDict[f"{emPlot_ax}_rotate"]:
            tabDict[f"{emPlot_ax}_em_data"][0].set_xdata(self.data.iloc[:, self.ex_ind])
        else:
            tabDict[f"{emPlot_ax}_em_data"][0].set_ydata(self.data.iloc[:, self.ex_ind])

        if tabDict[f"{exPlot_ax}_ex_data_ylabel"] is not None:
            tabDict[exPlot_ax].set_ylabel(f"Intensity (Em. {self.em_wavelengths[self.em_ind]} nm)")
        if tabDict[f"{emPlot_ax}_em_data_ylabel"] is not None:
            tabDict[emPlot_ax].set_ylabel(f"Intensity (Ex. {self.ex_wavelengths[self.ex_ind]} nm)")

        tabDict[emPlot_ax].relim()
        tabDict[emPlot_ax].autoscale_view()
        tabDict[exPlot_ax].relim()
        tabDict[exPlot_ax].autoscale_view()

        self.apply_em_plot_limits(tabDict, emPlot_ax)
        self.apply_ex_plot_limits(tabDict, exPlot_ax)
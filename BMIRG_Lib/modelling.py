import lmfit
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Modelling:
    def __init__(self):
        self.file = None
        self.model_init = None
        self.is_loaded = False
        self.em_wavelengths = None
        self.ex_wavelengths = None
        self.lmfit_model = None
        self.M0 = None
        self.data = None
        self.params = None
        self.adj_r2 = None
        self.eval_res = None
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

    def clear(self, tabDict, emPlot_ax, exPlot_ax, cont_ax):
        if f"{cont_ax}_cb" in tabDict:
            tabDict[f"{cont_ax}_cb"].remove()
        if f"{cont_ax}_em_ind" in tabDict:
            tabDict["blit_manager"].remove_artist(tabDict[f"{cont_ax}_em_ind"])
        if f"{cont_ax}_ex_ind" in tabDict:
            tabDict["blit_manager"].remove_artist(tabDict[f"{cont_ax}_ex_ind"])

        tabDict[cont_ax].clear()

        if f"{emPlot_ax}_em_model" in tabDict:
            tabDict[f"{emPlot_ax}_em_model"][0].remove()
        if f"{exPlot_ax}_ex_model" in tabDict:
            tabDict[f"{exPlot_ax}_ex_model"][0].remove()

    def open(self, file, mask):
        try:
            self.file = file
            self.model_init = pd.read_csv(file)
            self.is_loaded = True
            self.M0 = pd.read_csv(mask, header=None)
        except Exception as exc:
            raise Exception(f"Model or mask could not be loaded: {exc}")

    def generateModel(self):
        self.lmfit_model = None
        first_lineshape = None

        for i in range(self.model_init.shape[0]):
            if self.model_init.iloc[i, 6] == "yes":
                if self.model_init.iloc[i, 5] == "gaussian":
                    self.lmfit_model = lmfit.Model(
                        self.amp_gaussian_2d,
                        prefix=f"ls{i+1}_",
                        independent_vars=["xy_mesh", "mask"]
                    )
                else:
                    raise Exception("Unsupported lineshape. Only gaussian is currently supported.")
                first_lineshape = i
                break

        if self.lmfit_model is None:
            raise Exception("No active model components found in the model file.")

        for i in range(first_lineshape + 1, self.model_init.shape[0]):
            if self.model_init.iloc[i, 6] == "yes":
                if self.model_init.iloc[i, 5] == "gaussian":
                    self.lmfit_model += lmfit.Model(
                        self.amp_gaussian_2d,
                        prefix=f"ls{i+1}_",
                        independent_vars=["xy_mesh", "mask"]
                    )
                else:
                    raise Exception("Unsupported lineshape. Only gaussian is currently supported.")

    def amp_gaussian_2d(self, xy_mesh, mask, amp, xc, yc, xwidth, ywidth) -> np.ndarray:
        (x, y) = xy_mesh
        gauss = amp * np.exp(
            -((x - xc) ** 2 / (2 * xwidth ** 2) + (y - yc) ** 2 / (2 * ywidth ** 2))
        ) / (2 * np.pi * xwidth * ywidth)
        gauss *= mask
        return np.ravel(gauss)

    def set_ex_em_range(self, ex_wav, em_wav):
        self.ex_wavelengths = ex_wav
        self.em_wavelengths = em_wav

    def eval(self, data, xy_mesh):
        self.eval_res = self.lmfit_model.fit(
            np.ravel(data),
            self.params,
            xy_mesh=xy_mesh,
            mask=self.M0
        )
        self.adj_r2 = self.adj_rsquared()

    def generate_Params(self):
        self.params = lmfit.Parameters()
        for i in range(self.model_init.shape[0]):
            if self.model_init.iloc[i, 6] == "yes":
                self.params.add(f"ls{i+1}_amp", value=self.model_init.iloc[i, 0], vary=False)
                self.params.add(f"ls{i+1}_xc", value=self.model_init.iloc[i, 1], vary=False)
                self.params.add(f"ls{i+1}_yc", value=self.model_init.iloc[i, 2], vary=False)
                self.params.add(f"ls{i+1}_xwidth", value=self.model_init.iloc[i, 3], vary=False)
                self.params.add(f"ls{i+1}_ywidth", value=self.model_init.iloc[i, 4], vary=False)

    def adj_rsquared(self):
        k = 0
        for i in self.eval_res.params:
            if self.eval_res.params[i].vary is True:
                k += 1
        return 1 - (1 - self.rsquared()) * ((len(self.eval_res.data) - 1) / (len(self.eval_res.data) - (k + 1)))

    def rsquared(self):
        return 1 - self.eval_res.residual.var() / np.var(self.eval_res.data)

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
        self.data = self.eval_res.best_fit.reshape(len(self.em_wavelengths), len(self.ex_wavelengths))
        zdata = self.data.astype("float64")

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
            tabDict[ax].set_xlim(230, 810)

        if self.plot2d_yrange is not None:
            ymin, ymax = self.plot2d_yrange
            tabDict[ax].set_ylim(ymax, ymin)
        else:
            tabDict[ax].set_ylim(850, 280)

        tabDict[ax].set_yticks(np.arange(300, 901, 100))
        tabDict[ax].set_facecolor(self.facecolour)

        if axisLabels:
            tabDict[ax].set_xlabel("Excitation Wavelength (nm)")
            tabDict[ax].set_ylabel("Emission Wavelength (nm)")

        tabDict[f"{ax}_orig_loc"] = tabDict[ax].get_axes_locator()
        tabDict[f"{ax}_cax"] = make_axes_locatable(tabDict[ax]).append_axes("right", size="5%", pad=0.05)
        tabDict[f"{ax}_cb"] = tabDict["fig"].colorbar(psm, cax=tabDict[f"{ax}_cax"])
        tabDict[f"{ax}_cb"].set_label("Fluoresence")

        if title is not None:
            tabDict[ax].set_title(title)

    def emplot(self, tabDict, ax, colour="steelblue", rotate=False, invert_x_ax=False, invert_y_ax=False, x_label=None, y_label=None):
        if rotate:
            tabDict[f"{ax}_rotate"] = True
            tabDict[f"{ax}_em_model"] = tabDict[ax].plot(
                self.data[:, self.ex_ind],
                self.em_wavelengths,
                color=colour,
                linewidth=1
            )
            tabDict[ax].invert_xaxis()
        else:
            tabDict[f"{ax}_rotate"] = False
            tabDict[f"{ax}_em_model"] = tabDict[ax].plot(
                self.em_wavelengths,
                self.data[:, self.ex_ind],
                color=colour,
                linewidth=1
            )

        tabDict[f"{ax}_em_model_ylabel"] = y_label

        if invert_x_ax:
            tabDict[ax].invert_xaxis()
        if invert_y_ax:
            tabDict[ax].invert_yaxis()
        if x_label is not None:
            tabDict[ax].set_xlabel(x_label)
        if y_label is not None:
            tabDict[ax].set_ylabel(y_label)

        self.apply_em_plot_limits(tabDict, ax)

    def explot(self, tabDict, ax, colour="steelblue", x_label=None, y_label=None):
        tabDict[f"{ax}_ex_model"] = tabDict[ax].plot(
            self.ex_wavelengths,
            self.data[self.em_ind, :],
            color=colour,
            linewidth=1
        )
        tabDict[f"{ax}_ex_model_ylabel"] = y_label

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

    def refresh(self, data, xy_mesh, tabDict, emPlot_ax, exPlot_ax, cont_ax):
        self.open(self.file, "M0.csv")
        self.generateModel()
        self.generate_Params()
        self.eval(data, xy_mesh)

        if f"{cont_ax}_cb" in tabDict:
            tabDict[f"{cont_ax}_cb"].remove()
        tabDict[cont_ax].clear()

        self.plot(tabDict, cont_ax, title=f"Model Adj-R squared: {self.adj_r2:.2f}")
        self.indicators(tabDict, cont_ax, animated=True)

        tabDict[f"{exPlot_ax}_ex_model"][0].set_ydata(self.data[self.em_ind, :])
        if tabDict[f"{emPlot_ax}_rotate"]:
            tabDict[f"{emPlot_ax}_em_model"][0].set_xdata(self.data[:, self.ex_ind])
        else:
            tabDict[f"{emPlot_ax}_em_model"][0].set_ydata(self.data[:, self.ex_ind])

        self.apply_em_plot_limits(tabDict, emPlot_ax)
        self.apply_ex_plot_limits(tabDict, exPlot_ax)

    def update(self, tabDict, emPlot_ax, exPlot_ax, cont_ax):
        if self.data is None:
            return

        tabDict[f"{cont_ax}_em_ind"].set_xdata([self.ex_wavelengths[self.ex_ind]])
        tabDict[f"{cont_ax}_ex_ind"].set_ydata([self.em_wavelengths[self.em_ind]])
        tabDict[f"{exPlot_ax}_ex_model"][0].set_ydata(self.data[self.em_ind, :])

        if tabDict[f"{emPlot_ax}_rotate"]:
            tabDict[f"{emPlot_ax}_em_model"][0].set_xdata(self.data[:, self.ex_ind])
        else:
            tabDict[f"{emPlot_ax}_em_model"][0].set_ydata(self.data[:, self.ex_ind])

        if tabDict[f"{exPlot_ax}_ex_model_ylabel"] is not None:
            tabDict[exPlot_ax].set_ylabel(f"Intensity (Em. {self.em_wavelengths[self.em_ind]} nm)")
        if tabDict[f"{emPlot_ax}_em_model_ylabel"] is not None:
            tabDict[emPlot_ax].set_ylabel(f"Intensity (Ex. {self.ex_wavelengths[self.ex_ind]} nm)")

        tabDict[emPlot_ax].relim()
        tabDict[emPlot_ax].autoscale_view()
        tabDict[exPlot_ax].relim()
        tabDict[exPlot_ax].autoscale_view()

        self.apply_em_plot_limits(tabDict, emPlot_ax)
        self.apply_ex_plot_limits(tabDict, exPlot_ax)
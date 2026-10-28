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

        self.emission_offset_nm = 40.0

    def open(self, file, mask=None):
        try:
            self.file = file
            self.model_init = pd.read_csv(file)
            self.is_loaded = True
            self.M0 = None if mask is None else pd.read_csv(mask, header=None).to_numpy(dtype=float)
        except Exception as exc:
            raise Exception(f"Model could not be loaded: {exc}")

    def set_ex_em_range(self, ex_wav, em_wav):
        self.ex_wavelengths = np.asarray(ex_wav, dtype=float)
        self.em_wavelengths = np.asarray(em_wav, dtype=float)

    def build_mask_from_data(self, offset_nm=None):
        if self.ex_wavelengths is None or self.em_wavelengths is None:
            raise Exception("Excitation and emission wavelengths must be set before building M0.")
        if offset_nm is None:
            offset_nm = self.emission_offset_nm

        ex = np.asarray(self.ex_wavelengths, dtype=float)
        em = np.asarray(self.em_wavelengths, dtype=float)
        self.M0 = (em[:, None] >= (ex[None, :] + float(offset_nm))).astype(float)

    def generateModel(self):
        self.lmfit_model = None
        first_lineshape = None

        for i in range(self.model_init.shape[0]):
            active = str(self.model_init.iloc[i, 6]).strip().lower() == "yes"
            shape = str(self.model_init.iloc[i, 5]).strip().lower()
            if active:
                if shape != "gaussian":
                    raise Exception("Unsupported lineshape. Only gaussian is currently supported.")
                self.lmfit_model = lmfit.Model(
                    self.amp_gaussian_2d,
                    prefix=f"ls{i+1}_",
                    independent_vars=["xy_mesh", "mask"]
                )
                first_lineshape = i
                break

        if self.lmfit_model is None:
            raise Exception("No active model components found in the model file.")

        for i in range(first_lineshape + 1, self.model_init.shape[0]):
            active = str(self.model_init.iloc[i, 6]).strip().lower() == "yes"
            shape = str(self.model_init.iloc[i, 5]).strip().lower()
            if active:
                if shape != "gaussian":
                    raise Exception("Unsupported lineshape. Only gaussian is currently supported.")
                self.lmfit_model += lmfit.Model(
                    self.amp_gaussian_2d,
                    prefix=f"ls{i+1}_",
                    independent_vars=["xy_mesh", "mask"]
                )

    def amp_gaussian_2d(self, xy_mesh, mask, amp, xc, yc, xwidth, ywidth) -> np.ndarray:
        x, y = xy_mesh
        xwidth = max(float(xwidth), 1e-9)
        ywidth = max(float(ywidth), 1e-9)

        gauss = float(amp) * np.exp(
            -((x - float(xc)) ** 2 / (2 * xwidth ** 2) +
              (y - float(yc)) ** 2 / (2 * ywidth ** 2))
        ) / (2 * np.pi * xwidth * ywidth)

        gauss *= np.asarray(mask, dtype=float)
        return np.ravel(gauss)

    def generate_Params(self, data_for_scale=None):
        self.params = lmfit.Parameters()

        exmin = float(np.min(self.ex_wavelengths))
        exmax = float(np.max(self.ex_wavelengths))
        emmin = float(np.min(self.em_wavelengths))
        emmax = float(np.max(self.em_wavelengths))

        if data_for_scale is not None:
            data_arr = np.asarray(data_for_scale, dtype=float)
            amp_guess_max = max(float(np.nanmax(data_arr)), 1.0) * 1000.0
        else:
            amp_guess_max = 1e9

        for i in range(self.model_init.shape[0]):
            if str(self.model_init.iloc[i, 6]).strip().lower() != "yes":
                continue

            amp = float(self.model_init.iloc[i, 0])
            xc = float(self.model_init.iloc[i, 1])
            yc = float(self.model_init.iloc[i, 2])
            xw = float(self.model_init.iloc[i, 3])
            yw = float(self.model_init.iloc[i, 4])

            self.params.add(f"ls{i+1}_amp", value=amp, vary=True, min=0, max=amp_guess_max)
            self.params.add(f"ls{i+1}_xc", value=xc, vary=True, min=exmin, max=exmax)
            self.params.add(f"ls{i+1}_yc", value=yc, vary=True, min=emmin, max=emmax)
            self.params.add(f"ls{i+1}_xwidth", value=max(xw, 1.0), vary=True, min=1.0, max=max(exmax - exmin, 5.0))
            self.params.add(f"ls{i+1}_ywidth", value=max(yw, 1.0), vary=True, min=1.0, max=max(emmax - emmin, 5.0))

    def eval(self, data, xy_mesh):
        data_array = np.asarray(data, dtype=float)

        if self.M0 is None:
            self.M0 = np.ones_like(data_array, dtype=float)

        if self.M0.shape != data_array.shape:
            raise Exception(f"Mask shape {self.M0.shape} does not match data shape {data_array.shape}.")

        self.eval_res = self.lmfit_model.fit(
            np.ravel(data_array),
            self.params,
            xy_mesh=xy_mesh,
            mask=self.M0
        )
        self.adj_r2 = self.adj_rsquared()

    def rsquared(self):
        data_var = np.var(self.eval_res.data)
        if data_var == 0:
            return 1.0
        return 1 - self.eval_res.residual.var() / data_var

    def adj_rsquared(self):
        k = sum(1 for p in self.eval_res.params.values() if p.vary)
        n = len(self.eval_res.data)
        if n <= (k + 1):
            return self.rsquared()
        return 1 - (1 - self.rsquared()) * ((n - 1) / (n - (k + 1)))

    def _build_contour_levels(self, zdata):
        n_levels = max(2, int(self.contour_levels))
        if self.intensity_range is not None:
            vmin, vmax = self.intensity_range
        else:
            vmin = float(np.nanmin(zdata))
            vmax = float(np.nanmax(zdata))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return np.linspace(vmin, vmax, n_levels), vmin, vmax

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
            tabDict[ax].set_xlim(float(np.min(self.ex_wavelengths)), float(np.max(self.ex_wavelengths)))

        if self.plot2d_yrange is not None:
            ymin, ymax = self.plot2d_yrange
            tabDict[ax].set_ylim(ymax, ymin)
        else:
            tabDict[ax].set_ylim(float(np.max(self.em_wavelengths)), float(np.min(self.em_wavelengths)))

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
        self.open(self.file, None)
        self.generateModel()
        self.generate_Params(data_for_scale=data)
        self.build_mask_from_data(self.emission_offset_nm)
        self.eval(np.asarray(data, dtype=float), xy_mesh)

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
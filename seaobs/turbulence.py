import dask.delayed as delayed
import xarray as xr
import numpy as np
from .geometry import latlon2dydx


@delayed(pure=True)
def lagged_difference(a, dim, lag, order=1):
	return (a.shift(**{dim: -lag}) - a) ** order


@delayed(pure=True)
def norm(dx, dy):
	return xr.ufuncs.sqrt(dx ** 2 + dy ** 2) / 1.e3


def strfunc(data, dim, max_lag, vars=None, order=2, r_bins=None):
	"""
	Compute the nth order structure function on variables of a DataArray or a
	Dataset. DataArrays and Datasets must have `lon` and `lat` as coordinates
	for longitude and latitude.

	Parameters
	----------
	data : xarray.DataArray or xarray.Dataset
		The data containing the variables used to compute structure function
	dim : str
		The dimension along which the structure function is compute
	max_lag : int
		The maximum lag in number of points
	vars : str or list of str, optional
		The name of the variable to process. If the data input is a DataArray,
		the corresponding variable will be processed.
	order  : int, optional
		The order of the structure function to compute
	r_bins :  int or array of scalars
		If not None, the structure functions are averaged over these different
		bins

	Returns
	-------
	res : Dataset
		A dataset with the nothdifferent structure functions
	"""
	dy, dx = latlon2dydx(data['lat'], data['lon'], dim=dim)
	# Compute curvilinear coordinate
	s = norm(dx, dy).cumsum(dim=dim)
	output = xr.Dataset()
	dvar_dict = {}
	if vars is None:
		try:
			vars = [var for var in data.variables]
		# The above code line does not apply on a DataArray
		except AttributeError:
			vars = [data.name]
	elif isinstance(vars, str):
		vars = [vars]
	# Compute nth order structures functions on the different variables
	for var in vars:
		distance_list, delta_list = [], []
		for lag in range(1, max_lag):
			# Compute distance between points
			distance = lagged_difference(s, dim, lag)
			distance_list.append(distance)
			# Compute variable difference between points
			try:
				delta = lagged_difference(data[var], dim, lag, order=order)
			# The above code line does not apply on a DataArray
			except AttributeError:
				delta = lagged_difference(data, dim, lag, order=order)
			delta_list.append(delta.isel(**{dim: slice(None, -1)}))
		dvar = delayed(xr.concat)(delta_list, dim='lags').stack(r=(dim, 'lags'))
		dvar_dict['D%s%s' % (order, var)] = dvar.compute()
	# Store the results into a dataset
	output = output.assign(**dvar_dict)
	r = delayed(xr.concat)(distance_list, dim='lags').stack(r=(dim, 'lags'))
	output = output.assign_coords(r=r.compute())
	if r_bins is not None:
		output = avg_strfunc(output, r_bins)
		output = output.rename({'r_bins' : 'r'})
	return output


def avg_strfunc(dn, r_bins):
	"""
	Average the structure functions over different bins using
	:py:func:`xarray.Dataset.groupby_bins`

	Parameters
	----------
	dn : xarray.DataSet
		The Dataset containing the nth structure functions
	r_bins :  int or array of scalars
		The list of bins over which the structure functions are averaged (see
		:py:func:`xarray.Dataset.groupby_bins`)

	Returns
	-------
	res : xarray.DataSet
		The structure functions averaged
	"""
	r_labels = r_bins[1:] - np.diff(r_bins) / 2
	return dn.groupby_bins('r', r_bins, labels=r_labels).mean(dim='r')


def helmdec_on_strfunc(d2, ul='u', ut='v'):
	"""
	Perform a Helmholtz decomposition of the second order structure functions

	Parameters
	----------
	d2 : xarray.DataSet
		The Dataset containing the structure functions
	ul : str, optional
		The longitudinal velocity name. Default is 'u'.
	ut : str, optional
		The transverse velocity name, default is 'v'

	Returns
	-------
	res  :  xarray.DataSet
	The structure functions averaged
	"""
	d2l = 'D2%s' % ul
	d2t = 'D2%s' % ut
	from scipy.integrate import cumtrapz
	integral = cumtrapz((1. / d2['r_bins'] * (d2[d2t] - d2[d2l])),
	                     x=( d2['r_bins']), initial=0, axis=-1)
	d2r = d2[d2t] + integral
	d2d = d2[d2l] - integral
	return d2.assign(D2r=d2r, D2d=d2d)
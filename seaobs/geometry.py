import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS = 6371 * 1000


def latlon2yx(lat, lon):
    """
    Convert latitude and longitude arrays to y and x arrays in m

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates

    Returns
    -------
    y : array-like
        Zonal cartesian coordinates
    x : array-like
        Meridional cartesian coordinates
    """
    y = np.pi / 180. * EARTH_RADIUS * lat
    x = np.cos(np.pi / 180. * lat) * np.pi / 180. * EARTH_RADIUS * lon
    return y, x


def latlon2yx(lat, lon):
    """
    Convert latitude and longitude arrays to y and x arrays in m

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates

    Returns
    -------
    y : array-like
        Zonal cartesian coordinates
    x : array-like
        Meridional cartesian coordinates
    """
    y = np.pi / 180. * EARTH_RADIUS * lat
    x = np.cos(np.pi / 180. * lat) * np.pi / 180. * EARTH_RADIUS * lon
    return y, x


def latlon2dydx(lat, lon, dim, label='upper'):
    """
    Convert latitude and longitude arrays to elementary displacements in dy
    and dx

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates
    dim : str
        Dimension along which the differentiation is performed, generally
        associated with the time dimension.
    label : {'upper', 'lower'}, optional
        The new coordinate in dimension dim will have the values of
        either the minuend’s or subtrahend’s coordinate for values ‘upper’
        and ‘lower’, respectively.

    Returns
    -------
    dy : array-like
        Zonal elementary displacement in cartesian coordinates
    dx : array-like
        Meridional elementary displacement in cartesian coordinates
    """
    dlat = lat.diff(dim, label=label)
    dlon = lon.diff(dim, label=label)
    dy = np.pi / 180. * EARTH_RADIUS * dlat
    dx = np.cos(np.pi / 180. * lat) * np.pi / 180. * EARTH_RADIUS * dlon
    return dy, dx


def latlon2heading(lat, lon, dim, label='upper'):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    Parameters
    ----------

    Returns
    -------
      The bearing in degrees

    """
    dy, dx = latlon2dydx(lat, lon, dim, label=label)
    initial_heading = np.arctan2(dx, dy) * 180. / np.pi
    # Normalize the initial heading
    compass_heading = (initial_heading + 360) % 360
    return compass_heading


def lonlat2heading(lon, lat, dim, label='upper'):
    """
    """
    compass_heading = latlon2heading(lat, lon, dim, label=label)
    return compass_heading


def latlon2vu(lat, lon, dim):
	"""
	Estimate the  meriodional and zonal velocity based on the
	latitude and longitude coordinates.

	Paramaters
	----------
    lat : xarray.DataArray
        Latitudinal spherical coordinates
    lon : xarray.DataArray
        Longitudinal spherical coordinates
    dim : str
        Name of the time dimension.

	Returns
	------
	v : xarray.DataArray
		The meridional velocity
	u : xarray.DataArray
		The zonal velocity
	"""
	dy, dx = latlon2dydx(lat, lon)
	dt = pd.to_numeric(lat[dim].diff(dim=dim)) * 1e-9
	v, u = dx / dt, dy / dt
	return v, u


def lonlat2uv(lon, lat, dim):
	"""
	Estimate the  meriodional and zonal velocity based on the
	latitude and longitude coordinates.

	Paramaters
	----------
	lon : xarray.DataArray
        Longitudinal spherical coordinates
    lat : xarray.DataArray
        Latitudinal spherical coordinates
    dim : str
        Name of the time dimension.

	Returns
	------
	u : xarray.DataArray
		The zonal velocity
	v : xarray.DataArray
		The meridional velocity
	"""
	v, u = latlon2vu(lat, lon, dim)
	return u, v


def inpolygon(data, poly):
    """
    Mask the data outside a polygon using shapely. Data must have the
    longitudinal and latitudinal coordinates 'lon' and 'lat', respectively.

    Paramaters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to mask
    poly : BaseGeometry
        A polygon defined using shapely

    Returns
    -------
    res : xarray.DataArray or xarray.Dataset
        The data masked outside the poylgon
    """
    from shapely.geometry import Point
    lon = data['lon']
    lat =  data['lat']
    def inpolygon(polygon, xp, yp):
        return np.array([Point(x, y).intersects(polygon)
                         for x, y in zip(xp, yp)], dtype=np.bool)
    mask = inpolygon(poly, lon.data.ravel(), lat.data.ravel())
    da_mask = xr.DataArray(mask, dims=lon.dims, coords=lon.coords)
    return da_mask
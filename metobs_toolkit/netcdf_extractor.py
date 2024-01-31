#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:04:53 2024

@author: thoverga
"""
import os
import sys
import logging
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
import numpy as np
import netCDF4  # needed for readin netcdf

from metobs_toolkit.plotting_functions import (
    make_regular_fig,
    make_platcarree_fig,
    xarray_2d_plot,
)


logger = logging.getLogger(__name__)


class NetCDFExtractor:
    """Class holding methods for extracting data from Google Earth Engine."""

    def __init__(self, netcdf_file):

        # check if file exist
        assert os.path.isfile(netcdf_file), f"{netcdf_file} is not a file."
        self.netcdf_file = netcdf_file

        # Data holders
        self.ds = None

        # fmt Settings
        self.fmt_dims = {}
        self.fmt_proj = {}

    def __str__(self):
        if self.ds is None:
            return f"NetCDFExtractor for file: {self.netcdf_file}, no data imported."
        return f"NetCDFExtractor for file: {self.netcdf_file}"

    def __repr__(self):
        return str(self)

    def activate_pyfa_sfx(self):
        # format dimensions
        self.fmt_dims = {
            "y": "northening",
            "x": "eastening",
            # 'level': 'modellevel',
            "validate": "timestamp",
        }

        self.fmt_proj = {
            "is_latlon": True,
            "wkt_str": None,
            "wkt_str_from_attr": "projection",
        }

        # self.fmt_fields = {
        #     'SFX.T2M': {
        #                 'model_unit': 'Kelvin',
        #                 # 'description': 'Surfex t2m',
        #                 },
        #     'SFX.T2M_TEB': {
        #                     'model_unit': 'Kelvin',
        #                     # 'description': 'Surfex TEB t2m',
        #                     },
        #     'SFX.HU2M': {
        #                  'model_unit': 'percentage',
        #                  # 'description': 'Surfex humidty at 2m',
        #                  }
        #     }

    def import_data(self, **kwargs):

        logger.info(f"Importing {self} as netCDF")
        ds = xr.open_dataset(filename_or_obj=self.netcdf_file, **kwargs)

        mapped_dims = []  # all the dimension names that are mapped correctly

        # 1.format spatial dimensions and coordinates
        dims_inverse = {v: k for k, v in self.fmt_dims.items()}

        # test if northening and eastening are both mapped
        assert (
            "northening" in dims_inverse.keys()
        ), "northening dimension not found in fmt_settings"
        assert (
            "eastening" in dims_inverse.keys()
        ), "eastening dimension not found in fmt_settings"
        assert (
            dims_inverse["northening"] in ds.dims
        ), f"{dims_inverse['northening']} dimension not found in the data dimensions"
        assert (
            dims_inverse["eastening"] in ds.dims
        ), f"{dims_inverse['eastening']} dimension not found in the data dimensions"

        # rename to northening and eastening
        logger.debug(
            f"renaming spatial dims: {{dims_inverse['northening'] : 'northening', dims_inverse['eastening'] : 'eastening'}}"
        )
        ds = ds.rename(
            {
                dims_inverse["northening"]: "northening",
                dims_inverse["eastening"]: "eastening",
            }
        )
        mapped_dims.extend(["northening", "eastening"])

        # 2.format levels
        if "modellevel" in dims_inverse:
            assert (
                dims_inverse["modellevel"] in ds.dims
            ), f"{dims_inverse['modellevel']} dimension not found in the data dimensions"

            # rename to modellevel
            logger.debug(
                f"rename modellevels: {{dims_inverse['modellevel'] : 'modellevel'}}"
            )
            ds = ds.rename({dims_inverse["modellevel"]: "modellevel"})
            mapped_dims.append("modellevel")

        # 3. format time dimensions
        if "timestamp" in dims_inverse:
            assert (
                dims_inverse["timestamp"] in ds.dims
            ), f"{dims_inverse['timestamp']} dimension not found in the data dimensions"

            # rename to timestamp
            logger.debug(
                f"rename timestamps: {{dims_inverse['timestamp'] : 'timestamp'}}"
            )
            ds = ds.rename({dims_inverse["timestamp"]: "timestamp"})
            mapped_dims.append("timestamp")

        # # 4. Check the fields
        # present_variables = list(set(ds.variables) - set(ds.dims) - set(ds.coords))
        # for field in present_variables:
        #     if field not in self.fmt_fields.keys():
        #         print(f'WARNING: {field} not a knonw field! (skipped)')
        #         ds = ds.drop_vars(names=[field])

        # 5. Check projection
        ds.rio.set_spatial_dims("eastening", "northening", inplace=True)
        if self.fmt_proj["is_latlon"]:
            assert (
                abs(np.nanmax(ds["eastening"])) < 361
            ), f"The eastening coordinates cannot be longitudes : {ds['eastening']}"
            assert (
                abs(np.nanmax(ds["northening"])) < 361
            ), f"The northening coordinates cannot be latitudes : {ds['northening']}"
            wkt_str = "epsg:4326"
        else:
            if self.fmt_proj["wkt_str"] is not None:
                wkt_str = str(self.fmt_proj["wkt_str"])
            elif "wkt_str_from_attr" in self.fmt_proj.keys():
                wkt_str = str(ds.attrs[self.fmt_proj["wkt_str_from_attr"]])
            else:
                sys.exit("No projection info is given.")

        logger.debug(f"write {wkt_str} as crs.")
        ds.rio.write_crs(wkt_str, inplace=True)

        ds = reproject(ds=ds, target_epsg="EPSG:4326")
        mapped_dims.append("spatial_ref")

        # 5. Drop unuse dimensions
        for dimname in ds.dims:
            if dimname in mapped_dims:
                continue

            # This will be executed if dimname is an unknonw dimension
            if ds.sizes[dimname] == 1:
                logger.warning(
                    f"WARNING: the {dimname} dimension is unmapped and has a length of 1. This dimension will be removed."
                )
                ds = ds.drop_dims(dimname)

            else:
                sys.exit(
                    f"The {dimname} dimension is unmapped and is non-trivial (size > 1)."
                )
        self.ds = ds

    def extract_values_at_2d_fields(self, geoseries, method="nearest", tolerance=None):

        # subset to 2d fields
        d2variables = []
        for variable in self.get_present_variables():
            # if 'modellevel' not in self.data[field]:
            d2variables.append(variable)

        logger.debug(f"Subset to 2d fields: {d2variables}")
        ds2d = self.ds[d2variables]

        # reproject the geoseries to latlon
        reproj_series = geoseries.dropna().to_crs("epsg:4326")

        def _get_val(coord, arr, method, tolerance):
            return arr.sel(
                {"eastening": coord.x, "northening": coord.y},
                method=method,
                tolerance=tolerance,
            ).to_dataframe()

        df_series = reproj_series.apply(
            _get_val, arr=ds2d, method=method, tolerance=tolerance
        )

        # convert to a proper dataframe
        subdfs = []
        for station, modeldf in df_series.items():
            modeldf["name"] = station
            subdfs.append(modeldf)

        df = pd.concat(subdfs)
        df = df.reset_index()
        df = df.set_index(["name", "timestamp"])

        # comput distances between model points and observation points
        modelgeo = df[["eastening", "northening"]]
        modelgeo = modelgeo.droplevel("timestamp")
        modelgeo = modelgeo[~modelgeo.index.duplicated(keep="first")]
        modelgdf = gpd.GeoDataFrame(
            modelgeo,
            geometry=gpd.points_from_xy(modelgeo["eastening"], modelgeo["northening"]),
            crs="EPSG:4326",
        )

        def calc_metric_inter_distances(geoseries1, geoseries2):
            assert isinstance(
                geoseries1, type(gpd.GeoSeries())
            ), f"{geoseries1} is not a geopandas GeoSeries."
            assert isinstance(
                geoseries2, type(gpd.GeoSeries())
            ), f"{geoseries2} is not a geopandas GeoSeries."
            assert (
                geoseries1.index == geoseries2.index
            ).all(), "Indixes of both geoseries are not equal."

            geod = geoseries1.crs.get_geod()
            dist = geod.inv(
                geoseries1.x.to_numpy(),
                geoseries1.y.to_numpy(),
                geoseries2.x.to_numpy(),
                geoseries2.y.to_numpy(),
            )

            dist = pd.Series(index=geoseries1.index, data=dist[2])
            return dist

        modelgdf["tollerance_distances"] = calc_metric_inter_distances(
            geoseries1=modelgdf["geometry"], geoseries2=reproj_series
        )

        # Merge data and geoinfo
        df = df.reset_index()
        df = df.merge(modelgdf, how="left", left_on="name", right_index=True)

        # subset and order columns
        df = df.set_index(["name", "timestamp"])
        rel_columns = self.get_present_variables()
        rel_columns.append("tollerance_distances")
        df = df[rel_columns]

        # format datetimeindex to tz aware
        df = df.reset_index().set_index(["timestamp"])
        df.index = df.index.tz_localize(tz="UTC")
        df = df.reset_index().set_index(["name", "timestamp"])
        return df

    def get_present_variables(self):
        if self.ds is None:
            return []
        return list(set(self.ds.variables) - set(self.ds.dims) - set(self.ds.coords))

    def _is_latlon(self):
        if self.ds is None:
            return False
        if str(self.ds.rio.crs) == "EPSG:4326":
            return True
        else:
            return False

    def make_plot(
        self,
        variable,
        datetime=None,
        metadf=None,
        title=None,
        land=None,
        coastline=None,
        contour=False,
        contour_levels=10,
        **kwargs,
    ):

        assert not (self.ds is None), "Empty data."
        assert (
            variable in self.get_present_variables()
        ), f"{variable} not in known variables: {self.get_present_variables()}"

        xrplot = self.ds[variable]

        # # check if the field is 2D or 3D
        # if self._is_field_3d_spatial(fieldname=field):
        #     # a level must be specified
        #     if modellevel is None:
        #         # TODO
        #         sys.exit('fix this default 3d part')
        #     else:
        #         xrplot = xrplot.sel(modellevel=modellevel)

        # check which time instance
        if xrplot["timestamp"].data.shape[0] > 1:
            if datetime is None:
                # take the first intance by default.
                xrplot = xrplot.sel({"timestamp": xrplot["timestamp"].data[0]})
            else:
                dt = fmt_datetime(datetime)
                if dt not in xrplot["timestamp"].data:
                    sys.exit(
                        f'{dt} not found in the model timestamps: {xrplot["timestamp"].data}'
                    )
                xrplot = xrplot.sel({"timestamp": dt})

        # setup default values for coastline and land features
        islatlon = self._is_latlon()
        if land is None:
            if islatlon:
                land = True
            else:
                land = False

        if coastline is None:
            if islatlon:
                coastline = True
            else:
                coastline = False

        if (land) | (coastline):
            if not islatlon:
                sys.exit(
                    "Adding land and coastline features is only available in latlon coordinates"
                )
        if islatlon:
            fig, ax = make_platcarree_fig()
        else:
            fig, ax = make_regular_fig()

        # create title
        if title is None:
            title = f"{variable}"
            # description = self.data.attrs['_descriptions'][field]
            # if (description != str(None)) & (description != ""):
            #     title = f'{title} ({description})'
            # if 'modellevel' in xrplot.coords:
            #     lvl = xrplot['modellevel'].data
            #     title = f'{title} at modellevel: {lvl}'

            if "timestamp" in xrplot.coords:
                cur_timestamp = np.datetime_as_string(
                    xrplot["timestamp"].data, unit="s", timezone="UTC"
                )
                title = f"{title}, at {cur_timestamp}"

        ax = xarray_2d_plot(
            dxr=xrplot,
            ax=ax,
            title=title,
            grid=False,
            land=land,
            coastline=coastline,
            contour=contour,
            levels=contour_levels,
            # cbar_kwargs={'label': f'in {self.fmt_fields[variable]["model_unit"]}'},
            **kwargs,
        )

        return ax


def fmt_datetime(datetime):
    """convert datetime.datetime to np.datetime64"""
    return np.datetime64(datetime)


def reproject(ds, target_epsg="EPSG:4326", nodata=-999):
    """
    Reproject a Dataset to an other CRS by EPSG code.

    The x and y coordinates are transformed to a target EPSG code.


    Parameters
    ----------
    dataset : xarray.Dataset
        A Dataset with x and y coordinates and with a rio.crs attribute indicating
        the current projection.
    target_epsg : str, optional
        The CRS to project to in EPSG format. The default is 'EPSG:4326'.
    nodata : int, optional
        Numeric value for nodata (will be introduced when reprojecting). For
        some reason this value must be typecasted to integer ???
        The default is -999.

    Returns
    -------
    ds : xarray.Dataset
        The Dataset with x and y coords now in the target CRS.

    Note
    ------
    All 0-size dimensions (like datetime dimensions for FaDataset) are removed
    by the reprojection! So make shure to copy them over.

    """
    print(f"Reprojecting dataset to {target_epsg}.")
    # I am not a fan of -999 as nodata, but it must be a value that
    # can be typecast to integer (rasterio thing?)

    # dimension order must be that the last two dimensions are northening, eastening respecitvely
    target_dim_order = list(set(ds.dims) - set(["northening", "eastening"]))
    target_dim_order.extend(["northening", "eastening"])
    ds = ds.transpose(*target_dim_order)
    ds.rio.set_spatial_dims("eastening", "northening", inplace=True)
    for fieldname in list(ds.variables):
        if fieldname not in ds.dims:
            # only applicable on xarray, not on dataset
            ds[fieldname].rio.write_nodata(nodata, inplace=True)

    # ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.reproject(target_epsg, nodata=nodata)

    # remove no data
    ds = ds.where(ds != nodata)

    # rename the default x and y to northening and eastening
    ds = ds.rename({"x": "eastening", "y": "northening"})
    return ds

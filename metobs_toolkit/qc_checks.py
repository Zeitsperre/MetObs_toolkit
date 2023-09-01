#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:44:54 2022

@author: thoverga
"""

import sys
import pandas as pd
import numpy as np
import logging



from metobs_toolkit.df_helpers import (
    init_multiindex,
    init_multiindexdf,
    init_triple_multiindexdf,
    xs_save
)


logger = logging.getLogger(__name__)


try:
    import titanlib
except ModuleNotFoundError:
    logger.warning("Titanlib is not installed, install it manually if you want to use this functionallity.")

# =============================================================================
# Helper functions
# =============================================================================


def make_outlier_df_for_check(
    station_dt_list, obsdf, obstype, flag, stationname=None, datetimelist=None
):
    """
    V5
    Helper function to create an outlier dataframe for the given station(s) and datetimes. This will be returned by
    a quality control check and later added to the dastes.outlierdf.

    Multiple commum inputstructures can be handles

    A multiindex dataframe with the relevant observationtypes i.e. the values_in_dict and a specific quality flag column (i.g. the labels) is returned.

    Parameters

    station_dt_list : MultiIndex or list of tuples: (name, datetime)
        The stations with corresponding datetimes that are labeled as outliers.
    values_in_dict : Dictionary
        A Dictionary {columnname: pd.Series()} with the values on which this check is applied.
    flagcolumnname : String
        Name of the labels column
    flag : String
        The label for all the outliers.
    stationname : String, optional
        It is possible to give the name of one station. The default is None.
    datetimelist : DatetimeIndex or List, optional
        The outlier timestamps for the stationname. The default is None.

    Returns

    check_outliers : pd.Dataframe
        A multiindex (name -- datetime) datatframe with one column (columname) and all
        values are the flag.
    """

    if isinstance(station_dt_list, pd.MultiIndex):
        multi_idx = station_dt_list

    elif isinstance(station_dt_list, list):  # list of tuples: (name, datetime)
        multi_idx = pd.MultiIndex.from_tuples(
            station_dt_list, names=["name", "datetime"]
        )
    elif not isinstance(stationname, type(None)):
        if isinstance(datetimelist, pd.DatetimeIndex):
            datetimelist = datetimelist.to_list()
        if isinstance(datetimelist, list):
            indexarrays = list(zip([stationname] * len(datetimelist), datetimelist))
            multi_idx = pd.MultiIndex.from_tuples(
                indexarrays, names=["name", "datetime"]
            )

        else:
            sys.exit(f"Type of datetimelist: {type(datetimelist)} is not implemented.")

    # subset outliers
    outliersdf = obsdf.loc[multi_idx]

    # make the triple multiindex
    outliersdf["obstype"] = obstype
    outliersdf = outliersdf.set_index("obstype", append=True)

    # add flag
    outliersdf["label"] = flag

    # subset columns
    outliersdf = outliersdf[[obstype, "label"]].rename(columns={obstype: "value"})

    # replace values in obsdf by Nan
    obsdf.loc[multi_idx, obstype] = np.nan

    return obsdf, outliersdf


# =============================================================================
# Quality assesment checks on data import
# =============================================================================


def invalid_input_check(df, checks_info):
    checkname = "invalid_input"

    # fast scan wich stations and obstypes have nan outliers
    groups = (
        df.reset_index()
        .groupby("name")
        .apply(lambda x: (np.isnan(x).any()) & (np.isnan(x).all() == False))
    )

    # extract all obstype that have outliers
    outl_obstypes = groups.apply(lambda x: x.any(), axis=0)
    outl_obstypes = outl_obstypes[outl_obstypes].index.to_list()

    # first loop over the smallest sample: outlier obstypes
    outl_dict = {}

    for obstype in outl_obstypes:
        # get stations that have ouliers for this obstype
        outl_stations = groups.loc[groups[obstype], obstype].index.to_list()

        outl_multiidx = init_multiindex()
        for sta in outl_stations:
            # apply check per station
            outl_idx = (
                xs_save(df, sta, level="name", drop_level=False)[obstype]
                .isnull()
                .loc[lambda x: x]
                .index
            )
            outl_multiidx = outl_multiidx.append(outl_idx)

        outl_dict[obstype] = outl_multiidx

    # create outliersdf for all outliers for all osbtypes
    outl_df = init_multiindexdf()
    for obstype, outliers in outl_dict.items():
        df, specific_outl_df = make_outlier_df_for_check(
            station_dt_list=outliers,
            obsdf=df,
            obstype=obstype,
            flag=checks_info[checkname]["outlier_flag"],
        )
        outl_df = pd.concat([outl_df, specific_outl_df])

    return df, outl_df


def duplicate_timestamp_check(df, checks_info, checks_settings):
    """
    V3
    Looking for duplcate timestaps per station. Duplicated records are removed by the method specified in the qc_settings.

    Parameters

    df : pandas.DataFrame
        The observations dataframe of the dataset object (Dataset.df)

    Returns

    df : pandas.DataFrame()
        The observations dataframe updated for duplicate timestamps. Duplicated timestamps are removed.

    """

    checkname = "duplicated_timestamp"

    duplicates = pd.Series(
        data=df.index.duplicated(keep=checks_settings[checkname]["keep"]),
        index=df.index,
    )

    if not df.loc[duplicates].empty:
        logger.warning(
            f" Following records are labeld as duplicates: {df.loc[duplicates]}, and are removed"
        )

    # Fill the outlierdf with the duplicates
    outliers = df[df.index.duplicated(keep=checks_settings[checkname]["keep"])]

    # convert values to nan in obsdf
    for obstype in df.columns:
        df.loc[outliers.index, obstype] = np.nan

    # ------- Create a outliersdf -----------#
    # the 'make outliersdf' function cannont be use because of duplicated indices

    outliers = outliers.rename(
        columns={col: "value_" + col for col in outliers.columns}
    )
    outliers = outliers.reset_index()
    outliers["_to_get_unique_idx"] = np.arange(outliers.shape[0])

    outliersdf = pd.wide_to_long(
        df=outliers,
        stubnames="value",
        sep="_",
        suffix=r"\w+",  # to use non-integer suffexes
        i=["name", "datetime", "_to_get_unique_idx"],
        j="obstype",
    )
    # remove the temorary level from the index
    outliersdf = outliersdf.droplevel("_to_get_unique_idx", axis=0)

    # add label column
    outliersdf["label"] = checks_info[checkname]["outlier_flag"]

    # drop duplicates in the obsdf, because this gives a lot of troubles
    # The method does not really mater because the values are set to nan in the observations
    df = df[~df.index.duplicated(keep="first")]

    return df, outliersdf


# =============================================================================
# Quality assesment checks on dataset
# =============================================================================


def gross_value_check(obsdf, obstype, checks_info, checks_settings):
    """
    Looking for values of an observation type that are not physical. These values are labeled and the physical limits are specified in the qc_settings.

    Parameters

    input_series : pandas.Series
        The observations series of the dataset object

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'.


    Returns

    updated_obs_series : pandas.Series
        The observations series updated for this check. Observations that didn't pass are removed.

    outlier_df : pandas.DataFrame
        The collection of records flagged as outliers by this check.

    """

    checkname = "gross_value"

    try:
        specific_settings = checks_settings[checkname][obstype]
    except:
        logger.warning(
            f"No {checkname} settings found for obstype={obstype}. Check is skipped!"
        )
        return obsdf, init_multiindexdf()

    # drop outliers from the series (these are Nan's)
    input_series = obsdf[obstype].dropna()

    # find outlier observations as a list of tuples [(name, datetime), (name, datetime)]
    outl_obs = input_series.loc[
        (input_series <= specific_settings["min_value"])
        | (input_series >= specific_settings["max_value"])
    ].index.to_list()

    # make new obsdf and outlierdf
    obsdf, outlier_df = make_outlier_df_for_check(
        station_dt_list=outl_obs,
        obsdf=obsdf,
        obstype=obstype,
        flag=checks_info[checkname]["outlier_flag"],
    )

    return obsdf, outlier_df


def persistance_check(
    station_frequencies, obsdf, obstype, checks_info, checks_settings
):
    """
    V3
    Looking for values of an observation type that do not change during a timewindow. These are flagged as outliers.

    In order to perform this check, at least N observations chould be in that time window.


    Parameters

    input_series : pandas.Series
        The observations series of the dataset object

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'.


        Returns

        updated_obs_series : pandas.Series
            The observations series updated for this check. Observations that didn't pass are removed.

        outlier_df : pandas.DataFrame
            The collection of records flagged as outliers by this check.

    """

    checkname = "persistance"

    try:
        specific_settings = checks_settings[checkname][obstype]
    except:
        logger.warning(
            f"No {checkname} settings found for obstype={obstype}. Check is skipped!"
        )
        return obsdf, init_multiindexdf()

    invalid_windows_check_df = (
        pd.to_timedelta(specific_settings["time_window_to_check"]) / station_frequencies
        < specific_settings["min_num_obs"]
    )
    invalid_stations = list(
        invalid_windows_check_df[invalid_windows_check_df == True].index
    )
    if bool(invalid_stations):
        logger.warning(
            f"The windows are too small for stations  {invalid_stations} to perform persistance check"
        )

    subset_not_used = obsdf[obsdf.index.get_level_values("name").isin(invalid_stations)]
    subset_used = obsdf[~obsdf.index.get_level_values("name").isin(invalid_stations)]

    if not subset_used.empty:
        # drop outliers from the series (these are Nan's)
        input_series = subset_used[obstype].dropna()

        # apply persistance
        def is_unique(
            window,
        ):  # comp order of N (while using the 'unique' function is Nlog(N))
            a = window.values
            a = a[~np.isnan(a)]
            return (a[0] == a).all()

        # TODO: Tis is very expensive if no coarsening is applied !!!! Can we speed this up?
        window_output = (
            input_series.reset_index(level=0)
            .groupby("name")
            .rolling(
                window=specific_settings["time_window_to_check"],
                closed="both",
                center=True,
                min_periods=specific_settings["min_num_obs"],
            )
            .apply(is_unique)
        )

        list_of_outliers = []
        outl_obs = window_output.loc[window_output[obstype] == True].index
        for outlier in outl_obs:
            outliers_list = get_outliers_in_daterange(
                input_series,
                outlier[1],
                outlier[0],
                specific_settings["time_window_to_check"],
                station_frequencies,
            )

            list_of_outliers.extend(outliers_list)

        list_of_outliers = list(set(list_of_outliers))

        # make new obsdf and outlierdf
        subset_used, outlier_df = make_outlier_df_for_check(
            station_dt_list=list_of_outliers,
            obsdf=subset_used,
            obstype=obstype,
            flag=checks_info[checkname]["outlier_flag"],
        )

        obsdf = pd.concat([subset_used, subset_not_used])

        return obsdf, outlier_df

    else:
        obsdf = pd.concat([subset_used, subset_not_used])

        return obsdf, init_multiindexdf()


def repetitions_check(obsdf, obstype, checks_info, checks_settings):
    """
    Looking for values of an observation type that are repeated at least with the frequency specified in the qc_settings. These values are labeled.

    Parameters

    input_series : pandas.Series
        The observations series of the dataset object

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'.


    Returns

    updated_obs_series : pandas.Series
        The observations series updated for this check. Observations that didn't pass are removed.

    outlier_df : pandas.DataFrame
        The collection of records flagged as outliers by this check.


    """

    checkname = "repetitions"

    try:
        specific_settings = checks_settings[checkname][obstype]
    except:
        logger.warning(
            f"No {checkname} settings found for obstype={obstype}. Check is skipped!"
        )
        return obsdf, init_multiindexdf()

    # drop outliers from the series (these are Nan's)
    input_series = obsdf[obstype].dropna()

    # find outlier datetimes

    # add time interval between two consecutive records, group by consecutive records without missing records

    time_diff = input_series.index.get_level_values("datetime").to_series().diff()
    time_diff.index = input_series.index  # back to multiindex

    persistance_filter = ((input_series.shift() != input_series)).cumsum()

    grouped = input_series.groupby(["name", persistance_filter])
    # the above line groups the observations which have the same value and consecutive datetimes.
    group_sizes = grouped.size()
    outlier_groups = group_sizes[
        group_sizes > specific_settings["max_valid_repetitions"]
    ]

    # add to outl_obs.
    outl_obs = []
    for group_idx in outlier_groups.index:
        groupseries = grouped.get_group(group_idx)
        if len(set(groupseries)) == 1:  # Check if all observations are equal in group
            outl_obs.extend(groupseries.index.to_list())

    # make new obsdf and outlierdf
    obsdf, outlier_df = make_outlier_df_for_check(
        station_dt_list=outl_obs,
        obsdf=obsdf,
        obstype=obstype,
        flag=checks_info[checkname]["outlier_flag"],
    )

    return obsdf, outlier_df


def step_check(obsdf, obstype, checks_info, checks_settings):
    """

    V3
    Looking for jumps of the values of an observation type that are larger than the limit specified in the qc_settings. These values are removed from
    the input series and combined in the outlier df.

    The purpose of this check is to flag observations with a value that is too much different compared to the previous (not flagged) recorded value.

    Parameters

    input_series : pandas.Series
        The observations series of the dataset object

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'.



     Returns

     updated_obs_series : pandas.Series
         The observations series updated for this check. Observations that didn't pass are removed.

     outlier_df : pandas.DataFrame
         The collection of records flagged as outliers by this check.

    """

    checkname = "step"

    try:
        specific_settings = checks_settings[checkname][obstype]
    except:
        logger.warning(
            f"No {checkname} settings found for obstype={obstype}. Check is skipped!"
        )
        return obsdf, init_multiindexdf()

    # drop outliers from the series (these are Nan's)
    input_series = obsdf[obstype].dropna()

    list_of_outliers = []

    for name in input_series.index.droplevel("datetime").unique():
        subdata = xs_save(input_series, name, level="name", drop_level=False)

        time_diff = subdata.index.get_level_values("datetime").to_series().diff()
        time_diff.index = subdata.index  # back to multiindex
        # define filter
        step_filter = (
            (subdata - subdata.shift(1))
            > (
                specific_settings["max_increase_per_second"]
                * time_diff.dt.total_seconds()
            )
        ) | (
            (subdata - subdata.shift(1))
            < (
                specific_settings["max_decrease_per_second"]
                * time_diff.dt.total_seconds()
            )
        )  # &
        # (time_diff == station_frequencies[name]))
        outl_obs = step_filter[step_filter == True].index

        list_of_outliers.extend(outl_obs)

    # make new obsdf and outlierdf
    obsdf, outlier_df = make_outlier_df_for_check(
        station_dt_list=list_of_outliers,
        obsdf=obsdf,
        obstype=obstype,
        flag=checks_info[checkname]["outlier_flag"],
    )

    return obsdf, outlier_df


def window_variation_check(
    station_frequencies, obsdf, obstype, checks_info, checks_settings
):
    """

    V3
    Looking for jumps of the values of an observation type that are larger than the limit specified in the qc_settings. These values are removed from
    the input series and combined in the outlier df.

    There is a increament threshold (that is if there is a max value difference and the maximum value occured later than the minimum value occured.)
    And vice versa is there a decreament threshold.

    The check is only applied if there are at leas N observations in the time window.


    Parameters

    station_frequencies: pandas.Series
        The series containing the dataset time-resolution for all stations (as index).

    input_series : pandas.Series
        The observations series of the dataset object

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'.



     Returns

     updated_obs_series : pandas.Series
         The observations series updated for this check. Observations that didn't pass are removed.

     outlier_df : pandas.DataFrame
         The collection of records flagged as outliers by this check.

    """
    checkname = "window_variation"

    try:
        specific_settings = checks_settings[checkname][obstype]
    except:
        logger.warning(
            f"No {checkname} settings found for obstype={obstype}. Check is skipped!"
        )
        return obsdf, init_multiindexdf()

    invalid_windows_check_df = (
        pd.to_timedelta(specific_settings["time_window_to_check"]) / station_frequencies
        < specific_settings["min_window_members"]
    )
    invalid_stations = list(
        invalid_windows_check_df[invalid_windows_check_df == True].index
    )
    if bool(invalid_stations):
        logger.warning(
            f"The windows are too small for stations  {invalid_stations} to perform window variation check"
        )

    subset_not_used = obsdf[obsdf.index.get_level_values("name").isin(invalid_stations)]
    subset_used = obsdf[~obsdf.index.get_level_values("name").isin(invalid_stations)]

    if not subset_used.empty:
        # drop outliers from the series (these are Nan's)
        input_series = subset_used[obstype].dropna()

        # Calculate window thresholds (by linear extarpolation)
        windowsize_seconds = pd.Timedelta(
            specific_settings["time_window_to_check"]
        ).total_seconds()
        max_window_increase = (
            specific_settings["max_increase_per_second"] * windowsize_seconds
        )
        max_window_decrease = (
            specific_settings["max_decrease_per_second"] * windowsize_seconds
        )

        # apply steptest
        def variation_test(window):
            if (max(window) - min(window) > max_window_increase) & (
                window.idxmax() > window.idxmin()
            ):
                return 1

            if (max(window) - min(window) > max_window_decrease) & (
                window.idxmax() < window.idxmin()
            ):
                return 1
            else:
                return 0

        window_output = (
            input_series.reset_index(level=0)
            .groupby("name")
            .rolling(
                window=specific_settings["time_window_to_check"],
                closed="both",
                center=True,
                min_periods=specific_settings["min_window_members"],
            )
            .apply(variation_test)
        )

        list_of_outliers = []
        outl_obs = window_output.loc[window_output[obstype] == 1].index

        for outlier in outl_obs:
            outliers_list = get_outliers_in_daterange(
                input_series,
                outlier[1],
                outlier[0],
                specific_settings["time_window_to_check"],
                station_frequencies,
            )

            list_of_outliers.extend(outliers_list)

        list_of_outliers = list(set(list_of_outliers))

        # make new obsdf and outlierdf
        subset_used, outlier_df = make_outlier_df_for_check(
            station_dt_list=list_of_outliers,
            obsdf=subset_used,
            obstype=obstype,
            flag=checks_info[checkname]["outlier_flag"],
        )

        obsdf = pd.concat([subset_used, subset_not_used])

        return obsdf, outlier_df

    else:
        obsdf = pd.concat([subset_used, subset_not_used])

        return obsdf, init_multiindexdf()


def get_outliers_in_daterange(input_data, date, name, time_window, station_freq):
    end_date = date + (pd.Timedelta(time_window) / 2).floor(station_freq[name])
    start_date = date - (pd.Timedelta(time_window) / 2).floor(station_freq[name])

    daterange = pd.date_range(start=start_date, end=end_date, freq=station_freq[name])

    multi_idx = pd.MultiIndex.from_arrays(
        arrays=[[name] * len(daterange), daterange.to_list()],
        sortorder=1,
        names=["name", "datetime"],
    )
    outlier_sub_df = pd.DataFrame(data=None, index=multi_idx, columns=None)

    intersection = outlier_sub_df.index.intersection(input_data.dropna().index).values

    return intersection


# =============================================================================
# Titan checks
# =============================================================================

def create_titanlib_points_dict(obsdf, metadf, obstype):
    obs = obsdf[[obstype]]
    obs = obs.reset_index()

    # merge metadata
    obs = obs.merge(right = metadf[['lat', 'lon','altitude']],
                    how = 'left',
                    left_on='name',
                    right_index=True)

    dt_grouper = obs.groupby('datetime')


    points_dict = {}
    for dt, group in dt_grouper:

        check_group = group[~group[obstype].isnull()]

        points_dict[dt] = {
            'values': check_group[obstype].to_numpy(),
            'names': check_group['name'].to_numpy(),
            'lats': check_group['lat'].to_numpy(),
            'lons': check_group['lon'].to_numpy(),
            'elev': check_group['altitude'].to_numpy(),
            'ignore_names': group[group[obstype].isnull()]['name'].to_numpy()
        }

    return points_dict

# =============================================================================
# Toolkit buddy check
# =============================================================================




def _calculate_distance_matrix(metadf, metric_epsg='31370'):
    metric_metadf = metadf.to_crs(epsg=metric_epsg)
    return metric_metadf.geometry.apply(lambda g: metric_metadf.geometry.distance(g))



def _find_spatial_buddies(distance_df, buddy_radius):
    """ Get neighbouring stations using buddy radius."""
    buddies = {}
    for refstation, distances in distance_df.iterrows():
        bud_stations =distances[distances <= buddy_radius].index.to_list()
        bud_stations.remove(refstation)
        buddies[refstation] = bud_stations

    return buddies




# filter altitude buddies
def _filter_to_altitude_buddies(spatial_buddies, metadf, max_altitude_diff):
    """Filter neighbours by maximum altitude difference."""
    alt_buddies_dict = {}
    for refstation, buddylist in spatial_buddies.items():
        alt_diff = abs((metadf.loc[buddylist, 'altitude']) - metadf.loc[refstation, 'altitude'])
        alt_buddies = alt_diff[alt_diff <= max_altitude_diff].index.to_list()
        alt_buddies_dict[refstation] = alt_buddies
    return alt_buddies_dict





def _filter_to_samplesize(buddydict, min_sample_size):
    """Filter stations that are to isolated using minimum sample size."""
    to_check_stations = {}
    for refstation, buddies in buddydict.items():
        if len(buddies) < min_sample_size:
            # not enough buddies
            to_check_stations[refstation] = []  # remove buddies
        else:
            to_check_stations[refstation] = buddies
    return to_check_stations






def toolkit_buddy_check(obsdf, metadf, obstype, buddy_radius, min_sample_size, max_alt_diff,
                        min_std, std_threshold, outl_flag, metric_epsg='31370', lapserate=-0.0065):


    outliers_idx = init_multiindex()

    # Get spatial buddies for each station
    distance_df = _calculate_distance_matrix(metadf=metadf,
                                             metric_epsg=metric_epsg)
    buddies = _find_spatial_buddies(distance_df=distance_df,
                                    buddy_radius=buddy_radius)

    # Filter by altitude difference
    buddies = _filter_to_altitude_buddies(spatial_buddies=buddies,
                                          metadf=metadf,
                                          max_altitude_diff=max_alt_diff)

    # Filter by samplesize
    buddydict =_filter_to_samplesize(buddydict=buddies,
                                     min_sample_size=min_sample_size)

    # Apply buddy check station per station
    for refstation, buddies in buddydict.items():
        if len(buddies) == 0:
            logger.debug(f'{refstation} has not enough suitable buddies.')
            continue

        # Get observations
        buddies_obs = obsdf[obsdf.index.get_level_values('name').isin(buddies)][obstype]
        # Unstack
        buddies_obs = buddies_obs.unstack(level='name')

        # Make lapsrate correction:
        ref_alt = metadf.loc[refstation, 'altitude']
        buddy_correction = ((metadf.loc[buddies, 'altitude'] - ref_alt) * (-1. * lapserate)).to_dict()
        for bud in buddies_obs.columns:
            buddies_obs[bud] = buddies_obs[bud] - buddy_correction[bud]

        # calucalate std and mean row wise
        buddies_obs['mean'] = buddies_obs[buddies].mean(axis=1)
        buddies_obs['std'] = buddies_obs[buddies].std(axis=1)
        buddies_obs['samplesize'] = buddies_obs[buddies].count(axis=1)

        # from titan they use std adjust which is float std_adjusted = sqrt(variance + variance / n_buddies);
        # This is not used
        # buddies_obs['var'] = buddies_obs[buddies].var(axis=1)
        # buddies_obs['std_adj'] =np.sqrt(buddies_obs['var'] + buddies_obs['var']/buddies_obs['samplesize'])


        # replace where needed with min std
        buddies_obs['std'] = buddies_obs['std'].where(cond=buddies_obs['std'] >= min_std,
                                                      other=min_std)

        # Get refstation observations and merge
        ref_obs = obsdf[obsdf.index.get_level_values('name') == refstation][obstype].unstack(level='name')
        buddies_obs = buddies_obs.merge(ref_obs,
                                        how='left', #both not needed because if right, than there is no buddy sample per definition.
                                        left_index=True,
                                        right_index=True)
        # Calculate sigma
        buddies_obs['chi'] = (abs(buddies_obs['mean'] - buddies_obs[refstation])) / buddies_obs['std']

        outliers = buddies_obs[(buddies_obs['chi'] > std_threshold) & (buddies_obs['samplesize'] >= min_sample_size)]

        logger.debug(f' Buddy outlier details for {refstation}: \n {buddies}')
        # NOTE: the outliers (above) can be interesting to pass back to the dataset??


        # to multiindex
        outliers['name'] = refstation
        outliers = outliers.reset_index().set_index(['name', 'datetime']).index
        outliers_idx = outliers_idx.append(outliers)

    # Update the outliers and replace the obsdf

    obsdf, outlier_df = make_outlier_df_for_check(
        station_dt_list=outliers_idx,
        obsdf=obsdf,
        obstype=obstype,
        flag=outl_flag,
    )

    return obsdf, outlier_df



# =============================================================================
#
# =============================================================================
def titan_buddy_check(obsdf, metadf, obstype, checks_info, checks_settings, titan_specific_labeler):

    """
    The buddy check compares an observation against its neighbours (i.e. buddies). The check looks for
    buddies in a neighbourhood specified by a certain radius. The buddy check flags observations if the
    (absolute value of the) difference between the observations and the average of the neighbours
    normalized by the standard deviation in the circle is greater than a predefined threshold.


    Parameters

    obsdf: Pandas.DataFrame
        The dataframe containing the observations

    metadf: Pandas.DataFrame
        The dataframe containing the metadata (e.g. latitude, longitude...)

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'

    checks_info: Dictionary
        Dictionary with the names of the outlier flags for each check

    checks_settings: Dictionary
        Dictionary with the settings for each check

    titan_specific_labeler: Dictionary
        Dictionary that maps numeric flags to 'ok' or 'outlier' flags for each titan check


    """

    try:
        alt = metadf['altitude']
    except:
        logger.warning(
            f"Cannot find altitude of weather stations. Check is skipped!"
        )

    # Create points_dict
    pointsdict = create_titanlib_points_dict(obsdf, metadf, obstype)

    df_list = []
    for dt, point in pointsdict.items():
        obs = list(point['values'])
        titan_points = titanlib.Points(np.asarray(point['lats']),
                                       np.asarray(point['lons']),
                                       np.asarray(point['elev']))


        num_labels = titanlib.buddy_check(
                titan_points,
                np.asarray(obs),
                np.asarray([checks_settings['radius']] * len(obs)), #same radius for all stations
                np.asarray([checks_settings['num_min']] * len(obs)), #same min neighbours for all stations
                checks_settings['threshold'],
                checks_settings['max_elev_diff'],
                checks_settings['elev_gradient'],
                checks_settings['min_std'],
                checks_settings['num_iterations'],
                np.full(len(obs), 1)) #check all

        labels = pd.Series(num_labels, name='num_label').to_frame()
        labels['name'] = point['names']
        labels['datetime'] = dt
        df_list.append(labels)

    checkeddf = pd.concat(df_list)

    #Convert to toolkit format
    outliersdf = checkeddf[checkeddf['num_label'].isin(titan_specific_labeler['outl'])]

    outliersdf = outliersdf.set_index(['name', 'datetime'])


    obsdf, outliersdf = make_outlier_df_for_check(station_dt_list=outliersdf.index,
                                                  obsdf = obsdf,
                                                  obstype=obstype,
                                                  flag = checks_info["titan_buddy_check"]['outlier_flag'])


    return obsdf, outliersdf

def titan_sct_resistant_check(obsdf, metadf, obstype,
                              checks_info, checks_settings,
                              titan_specific_labeler):

    """
    The SCT resistant check is a spatial consistency check which compares each observations to what is expected given the other observations in the
    nearby area. If the deviation is large, the observation is removed. The SCT uses optimal interpolation
    (OI) to compute an expected value for each observation. The background for the OI is computed from
    a general vertical profile of observations in the area.

    Parameters

    obsdf: Pandas.DataFrame
        The dataframe containing the observations

    metadf: Pandas.DataFrame
        The dataframe containing the metadata (e.g. latitude, longitude...)

    obstype: String, optional
        The observation type that has to be checked. The default is 'temp'

    checks_info: Dictionary
        Dictionary with the names of the outlier flags for each check

    checks_settings: Dictionary
        Dictionary with the settings for each check

    titan_specific_labeler: Dictionary
        Dictionary that maps numeric flags to 'ok' or 'outlier' flags for each titan check


    """

    import time

    try:
        alt = metadf['altitude']
    except:
        logger.warning(
            f"Cannot find altitude of weather stations. Check is skipped!"
        )

    # Create points_dict
    pointsdict = create_titanlib_points_dict(obsdf, metadf, obstype)

    df_list = []
    for dt, point in pointsdict.items():
        logger.debug(f'sct on observations at {dt}')
        obs = list(point['values'])
        titan_points = titanlib.Points(np.asarray(point['lats']),
                                       np.asarray(point['lons']),
                                       np.asarray(point['elev']))


        flags, scores = titanlib.sct_resistant(
                points=titan_points, #points
                values=np.asarray(obs), # vlues
                obs_to_check=np.full(len(obs), 1), # obs to check (check all)
                background_values=np.full(len(obs), 0), # background values
                background_elab_type=titanlib.MedianOuterCircle, #background elab type
                num_min_outer=checks_settings['num_min_outer'], #num min outer
                num_max_outer=checks_settings['num_max_outer'], #num mac outer
                inner_radius=checks_settings['inner_radius'], #inner radius
                outer_radius=checks_settings['outer_radius'], #outer radius
                num_iterations=checks_settings['num_iterations'], #num iterations
                num_min_prof=checks_settings['num_min_prof'], #num min prof
                min_elev_diff=checks_settings['min_elev_diff'], # min elev diff
                min_horizontal_scale=checks_settings['min_horizontal_scale'], #min horizontal scale
                max_horizontal_scale=checks_settings['max_horizontal_scale'], # max horizontal scale
                kth_closest_obs_horizontal_scale=checks_settings['kth_closest_obs_horizontal_scale'], #kth closest obs horizontal scale
                vertical_scale=checks_settings['vertical_scale'], #vertical scale
                value_mina=[x - checks_settings['mina_deviation'] for x in obs], # values mina
                value_maxa=[x + checks_settings['maxa_deviation'] for x in obs], #values maxa
                value_minv=[x - checks_settings['minv_deviation'] for x in obs], #values minv
                value_maxv=[x + checks_settings['maxv_deviation'] for x in obs], # values maxv
                eps2=np.full(len(obs),checks_settings['eps2']), #eps2
                tpos=np.full(len(obs),checks_settings['tpos']), #tpos
                tneg=np.full(len(obs),checks_settings['tneg']), #tneg
                debug=checks_settings['debug'], #debug
                basic=checks_settings['basic'] #basic
                )
        logger.debug('Sleeping ... (to avoid segmentaton errors)')
        time.sleep(1)

        labels = pd.Series(flags, name='num_label').to_frame()
        labels['name'] = point['names']
        labels['datetime'] = dt
        df_list.append(labels)

    checkeddf = pd.concat(df_list)

    #Convert to toolkit format
    outliersdf = checkeddf[checkeddf['num_label'].isin(titan_specific_labeler['outl'])]

    outliersdf = outliersdf.set_index(['name', 'datetime'])


    obsdf, outliersdf = make_outlier_df_for_check(station_dt_list=outliersdf.index,
                                                  obsdf = obsdf,
                                                  obstype=obstype,
                                                  flag = checks_info["titan_sct_resistant_check"]['outlier_flag'])


    return obsdf, outliersdf



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:00:45 2024

@author: thoverga
"""


import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from metobs_toolkit.obstypes import Obstype
from metobs_toolkit.analysis import _make_time_derivatives

from metobs_toolkit.df_helpers import get_seasons, _make_time_derivatives

# from metobs_toolkit.dataset import Dataset
# from metobs_verif.verification_methods import get_basic_scores_dict
# from metobs_verif.modeloutput import Modelfield
# import metobs_verif.plotting as plotting

logger = logging.getLogger(__name__)


class Verification:
    def __init__(self, modeldata, dataset):
        self.modeldata = modeldata
        self.dataset = dataset  # dataset or station

        self.verifdf = None
        self.loc_to_gp_distance = None

        # define plot defaults
        self.modelcolor = "blue"
        self.obscolor = "orange"
        self.biascolor = "#D5573B"
        self.rmsecolor = "#885053"
        self.maecolor = "#777DA7"
        self.corcolor = "#94C9A9"

        # self._construct_verifdf()
        self._check_compatibility()

    def __repr__(self):
        return f"Verification object of \n {self.dataset} \n ------------- And ---------------- \n {self.modeldata} "

    def __str__(self):
        return f"Verification object of \n {self.dataset} \n ------------- And ---------------- \n {self.modeldata} "

    def _check_compatibility(self):
        # test location and names are equal between dataset and modeldat
        assert not (self.modeldata.df.empty), f"The Modeldata is empty."
        assert not (self.dataset.df.empty), f"The Dataset is empty."
        assert not (
            self.modeldata.metadf.empty
        ), f"The metadf of the modeldata is empty"
        assert not (self.dataset.metadf.empty), f"The metadf of the modeldata is empty"

        startobs = self.dataset.df.index.get_level_values("datetime").min()
        endobs = self.dataset.df.index.get_level_values("datetime").max()

        startmod = self.modeldata.df.index.get_level_values("datetime").min()
        endmod = self.modeldata.df.index.get_level_values("datetime").max()
        assert (
            startobs < endmod
        ), f"Start of observations {startobs} is after the end of the modeldata {endmod}"
        assert (
            startmod < endobs
        ), f"Start of modeldata {startmod} is after the end of the modeldata {endobs}"

        assert set(self.modeldata.metadf.index) == set(
            self.dataset.metadf.index
        ), f"The stationnames are not equal between the Modeldata and the Dataset."
        assert (
            self.modeldata.metadf["lat"] == self.dataset.metadf["lat"]
        ).all(), "The coordinates of the stations are not equal between the observations and the modeldata."
        assert (
            self.modeldata.metadf["lon"] == self.dataset.metadf["lon"]
        ).all(), "The coordinates of the stations are not equal between the observations and the modeldata."

    # def get_obs_obstypes(self):
    #     #retunr list
    #     all_obstypes = self.dataset.obstypes
    #     return [val for key, val in all_obstypes.items() if key in self.dataset.df.columns]

    # def get_model_fields(self):
    #     return self.modeldata.get_fields()

    # def _get_model_field_obstypes_dict(self):
    #     modelfields = self.get_model_fields()
    #     return {field: self.modeldata.data.attrs['_obstypes'][field] for field in modelfields}

    # def _construct_verifdf(self):
    #     # Test the observations
    #     assert not self.dataset.metadf.empty, f'{self.obs} has an empty metadf attribute.'
    #     assert isinstance(self.dataset.metadf, type(gpd.GeoDataFrame())), 'the observations metadf attribute is not a GeoDataFrame.'

    #     modeldf = self.modeldata.extract_values_at_2d_fields(geoseries = self.dataset.metadf['geometry'])

    #     # Merge observations and modeldata
    #     obsdf = self.dataset.df.reset_index()
    #     #TODO add time tollerance???
    #     verifdf= obsdf.merge(modeldf,
    #                           how='inner',
    #                           left_on=['name', 'datetime'],
    #                           right_on=['name', 'timestamp'])
    #     if verifdf.empty:
    #         sys.exit('No overlap found for the observations and the model point extractions.')

    #     # create a loc_to_gp_distance series
    #     self.loc_to_gp_distance = pd.Series(dict(zip(verifdf['name'], verifdf['tollerance_distances'])))
    #     verifdf = verifdf.drop(columns=['tollerance_distances'])

    #     verifdf = verifdf.reset_index(drop=True).set_index(['name', 'datetime'])
    #     # sort obstypes
    #     colslist = list(self.dataset.df.columns)
    #     colslist.extend(self.get_model_fields())
    #     verifdf = verifdf[colslist]
    #     self.verifdf = verifdf

    # def _get_corresponding_model_var(self, obstype):

    #     if isinstance(obstype, str):
    #         obstypestr = obstype
    #     elif isinstance(obstype, Obstype):
    #         obstypestr = obstype.name
    #     else:
    #         sys.exit(f'{obstype} not a string or Obstype.')

    #     field_obstype_dict = self._get_model_field_obstypes_dict()
    #     model_variabels = [key for key, val in field_obstype_dict.items() if val.name == obstypestr]
    #     assert len(model_variabels) > 0, f'There are no fields found with {observation_obstype} as a unit: {field_obstype_dict}'
    #     return model_variabels

    # # def get_point_verif(self, observation_obstype, model_variabels=None):
    # #     # check if obstype exist
    # #     present_obstypes = [obstype.name for obstype in self.get_obs_obstypes()]
    # #     assert observation_obstype in present_obstypes, f'{observation_obstype} not in the knonw obstypes: {present_obstypes}'

    # #     if isinstance(model_variabels, str):
    # #         model_variabels = [model_variabels]
    # #     if model_variabels is None:
    # #         #get all fields in the same observationtype
    # #         model_variabels = self._get_corresponding_model_var(obstype=observation_obstype)

    # #     #Check if all model variables are known
    # #     if not np.all([var in self.verifdf.columns for var in model_variabels]):
    # #         sys.exit(f'No all {model_variabels} are found in the verificatin table columns: {self.verifdf.columns}')

    # #     for var in model_variabels:
    # #         assert var in self.verifdf.columns, f'{var} not found in the verification table columns: {self.verifdf.columns}'

    # #     # COmpute basic scores
    # #     scores = {}
    # #     obstype = self.dataset.obstypes[observation_obstype]

    # #     fig, axs = plotting._make_plot_score_grid_axes()
    # #     fig.suptitle(self.modeldata._get_duration_representation_str(), fontsize=16)
    # #     for var in model_variabels:
    # #         fig, axs = plotting._make_plot_score_grid_axes()
    # #         fig.suptitle(f'{var} point verif for {self.modeldata._get_duration_representation_str()}', fontsize=10)

    # #         scores[var] = get_basic_scores_dict(model=self.verifdf[var],
    # #                                             obs=self.verifdf[observation_obstype])
    # #         axs[0] = plotting.plot_table(data=pd.Series(scores[var]),
    # #                                    ax=axs[0])
    # #         # Make scatter
    # #         axs[1] = plotting.simple_scatter_plot(data_x = self.verifdf[observation_obstype].values,
    # #                                      data_y = self.verifdf[var].values,
    # #                                      ax=axs[1],
    # #                                      y_label=f'{var} in {obstype.std_unit}',
    # #                                      x_label=f'{observation_obstype} in {obstype.std_unit}')

    # #         # Make histogram cmpariosn
    # #         axs[2] = plotting.comparison_hist_plot(df=self.verifdf[[obstype.name, var]].reset_index(drop=True),
    # #                                             ax=axs[2],
    # #                                             col_map_dict = {obstype.name: self.obscolor,
    # #                                                             var: self.modelcolor},
    # #                                             xlabel = f'{obstype.name} in {obstype.std_unit}',
    # #                                             ylabel = 'Frequency',
    # #                                             title='',
    # #                                             bins='auto',
    # #                                             orientation='vertical')

    def get_verification_analysis(
        self, observation_obstype="temp", model_obstype="temp_sfx", groupby=[""]
    ):

        # Check if obstypes exists
        assert (
            observation_obstype in self.dataset.obstypes.keys()
        ), f"{observation_obstype} not found in the known observational obstypes: {self.dataset.obstypes}"
        assert (
            model_obstype in self.modeldata.obstypes.keys()
        ), f"{model_obstype} not found in the known model obstypes: {self.modeldata.obstypes}"

        # Check if both obstype have thes ame standard unit
        assert (
            self.dataset.obstypes[observation_obstype].get_standard_unit()
            == self.modeldata.obstypes[model_obstype].get_standard_unit()
        ), f"The standard units of {observation_obstype} is not equal to {model_obstype}"

        # if isinstance(model_variabels, str):
        #     model_variabels = [model_variabels]
        # if model_variabels is None:
        #     #get all fields in the same observationtype
        #     model_variabels = self._get_corresponding_model_var(obstype=observation_obstype)

        scoringdf = self._get_grouped_point_scoring_metrics(
            observation_obstype=self.dataset.obstypes[observation_obstype],
            model_obstype=self.modeldata.obstypes[model_obstype],
            groupby=groupby,
        )
        return scoringdf
        # Find out which situation is applicable
        # cur_Obstype = self.dataset.obstypes[observation_obstype]

        # #aggregated plot (thus over all groups if defined)
        # add_scoring_var_plot = False
        # if len(set(scoringdf.index.names)) > 1:
        #     add_scoring_var_plot = True

        # fig, axdict = plotting._make_verif_grid(variables=model_variabels,
        #                                         add_extra_row=add_scoring_var_plot)

    #     # Plot table
    #     tabledf = scoringdf.reset_index().copy()
    #     numcols = ['bias','RMSE', 'MAE', 'cor']
    #     for col in numcols:
    #         tabledf[col] = tabledf[col].apply(lambda x: f'{x:.3f}')
    #     plotting.plot_table(data=tabledf,
    #                         ax=axdict['scoringtable'])

    #     for var in model_variabels:
    #         #scatter plots
    #         axdict[var]['scatter'] = plotting.simple_scatter_plot(data_x = self.verifdf[observation_obstype].values,
    #                                              data_y = self.verifdf[var].values,
    #                                              ax=axdict[var]['scatter'],
    #                                              y_label=f'{var} in {cur_Obstype.std_unit}',
    #                                              x_label=f'{cur_Obstype.name} in {cur_Obstype.std_unit}')

    #         #Histogram plots
    #         axdict[var]['hist'] = plotting.comparison_hist_plot(df=self.verifdf[[cur_Obstype.name, var]].reset_index(drop=True),
    #                                                ax=axdict[var]['hist'],
    #                                                col_map_dict = {cur_Obstype.name: self.obscolor,
    #                                                                var: self.modelcolor},
    #                                                xlabel = f'{cur_Obstype.name} in {cur_Obstype.std_unit}',
    #                                                ylabel = 'Frequency',
    #                                                title='',
    #                                                bins='auto',
    #                                                orientation='vertical')

    #     # 1: Situation 1 : Only variables in index --> no category/time evolution

    #     # 2: Situation 2 : only variables and datetime in index -->
    #     if len(set(scoringdf.index.names)) > 1 :
    #         print('time evolving situation')

    #         for var in model_variabels:
    #             plotdf = scoringdf.xs(var, level='variabel').drop(columns=['N_verifpoints'])

    #             axdict[var]['score_var'] = plotting.simple_multiline_plot(df=plotdf,
    #                                                 ax=axdict[var]['score_var'],
    #                                                 col_map_dict = {'bias': self.biascolor,
    #                                                                 'RMSE': self.rmsecolor,
    #                                                                 'MAE': self.maecolor,
    #                                                                 'cor': self.corcolor},
    #                                                 xlabel= f'{list(plotdf.index.names)}',
    #                                                 ylabel='',
    #                                                 add_zero=True)

    #             # return ax

    #     # # 3: Situation 3 : Variables and categorical levels in index --->
    #     # else:
    #     #     print('general situation')

    def _construct_mod_obs_df(self, obs_obstype, mod_obstype, interp=True):

        obsdf = self.dataset.df[[obs_obstype.name]]

        # TODO, without interpolation
        if interp:
            mod_df = self.modeldata.sample_data_as(target=obsdf)
            mod_df = mod_df[[mod_obstype.name]]
        else:
            sys.exit("not implemented yet")

        combdf = mod_df.merge(obsdf, how="outer", left_index=True, right_index=True)
        return combdf

    def _get_grouped_point_scoring_metrics(
        self, observation_obstype, model_obstype, groupby=[""]
    ):
        # check if obstype exist
        # present_obstypes = [obstype.name for obstype in self.get_obs_obstypes()]
        # assert observation_obstype in present_obstypes, f'{observation_obstype} not in the knonw obstypes: {present_obstypes}'

        # format groupby
        if groupby == [""]:
            groupby = None
        if groupby is not None:
            assert np.array(
                [
                    grp_id
                    in [
                        "minute",
                        "hour",
                        "month",
                        "year",
                        "day_of_year",
                        "week_of_year",
                        "season",
                        "datetime",
                        "name",
                        "lcz",
                    ]
                    for grp_id in groupby
                ]
            ).all(), f"Unknonw groupid in {groupby}."

        # # format model variables
        # if isinstance(model_variabels, str):
        #     model_variabels = [model_variabels]
        # if model_variabels is None:
        #     #get all fields in the same observationtype
        #     model_variabels = self._get_corresponding_model_var(obstype=observation_obstype)

        # #Check if all model variables are known
        # if not np.all([var in self.verifdf.columns for var in model_variabels]):
        #     sys.exit(f'No all {model_variabels} are found in the verificatin table columns: {self.verifdf.columns}')
        # for var in model_variabels:
        #     assert var in self.verifdf.columns, f'{var} not found in the verification table columns: {self.verifdf.columns}'

        # Construct dataframe to calculate grouped scores on
        # relevant_columns =
        # relevant_columns = model_variabels.copy()
        # relevant_columns.append(observation_obstype)
        # df = self.verifdf[relevant_columns].reset_index().rename(columns={'validate': 'datetime'})

        df = self._construct_mod_obs_df(
            obs_obstype=observation_obstype, mod_obstype=model_obstype, interp=True
        )
        df = df.reset_index()
        # add time aggregated columns
        df = _make_time_derivatives(df=df, required="", get_all=True)

        # get lcz for stations
        lcz_mapper = self.dataset.metadf["lcz"].to_dict()
        df["lcz"] = df["name"].map(lcz_mapper)

        # rename validate
        # df = df.rename(columns={'datetime': 'validate'})

        # subset the dataframe
        relevant_columns = [observation_obstype.name, model_obstype.name]
        if groupby is not None:
            relevant_columns.extend(groupby)

        df = df[relevant_columns]

        # Compute scores per group
        scoringlist = []
        if groupby is not None:
            # for var in model_variabels:
            for idx, group in df.groupby(groupby):
                groupscores = get_basic_scores_dict(
                    model=group[model_obstype.name], obs=group[observation_obstype.name]
                )
                # convert scores to a dataframe
                groupscores.update({"group": idx})
                groupscores.update({"variabel": model_obstype.name})

                groupscoresdf = pd.Series(groupscores).to_frame().transpose()
                trg_index = ["group", "variabel"]
                groupscoresdf = groupscoresdf.set_index(trg_index)
                scoringlist.append(groupscoresdf)
        if groupby is None:
            # for var in model_variabels:
            groupscores = get_basic_scores_dict(
                model=df[model_obstype.name], obs=df[observation_obstype.name]
            )

            # convert scores to a dataframe
            groupscores.update({"variabel": model_obstype.name})
            groupscoresdf = pd.Series(groupscores).to_frame().transpose()
            groupscoresdf = groupscoresdf.set_index("variabel")
            scoringlist.append(groupscoresdf)

        if not bool(scoringlist):
            sys.exit(f"No groups could be made for {groupby}")
        scoringdf = pd.concat(scoringlist)
        return scoringdf


def get_basic_scores_dict(model, obs):
    tot_scores = {}
    tot_scores["bias"] = _calc_bias(model, obs)
    tot_scores["N_verifpoints"] = (model - obs).dropna().shape[0]
    tot_scores["RMSE"] = _calc_rmse(model, obs)
    tot_scores["MAE"] = _calc_mae(model, obs)
    tot_scores["cor"] = _calc_cor(model, obs)
    return tot_scores


def _calc_bias(model, obs):
    return (model - obs).mean(skipna=True)


def _calc_rmse(model, obs):
    return np.sqrt(np.mean((model - obs) ** 2))


def _calc_mae(model, obs):
    return (model - obs).abs().mean(skipna=True)


def _calc_cor(model, obs):
    return model.corr(obs)

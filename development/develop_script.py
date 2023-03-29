#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:25:02 2022
@author: thoverga
"""

#%%
import vlinder_toolkit
import os
import sys
from pathlib import Path


lib_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(lib_folder))




#%% % Import


testdatafile = os.path.join(
    str(lib_folder), 'tests', 'test_data',  'testdata_okt_small.csv')

static_data = os.path.join(
    str(lib_folder), 'static_data', 'vlinder_metadata.csv')



# #% Setup dataset



dataset = vlinder_toolkit.Dataset()
dataset.update_settings(input_data_file=testdatafile,
                        input_metadata_file=static_data,
                        output_folder='/home/thoverga/Documents/VLINDER_github/vlinder_toolkit'
                        )

# dataset.apply_quality_control()


dataset.import_data_from_file(coarsen_timeres=True)
# dataset.apply_quality_control()





# dataset.set_timezone()

# # dataset.import_data_from_file(coarsen_timeres=True)

# # dataset.apply_quality_control()
# # dataset.get_qc_stats()
# dataset.make_geo_plot()
# dataset.make_plot()


# dataset.make_plot(stationnames=['vlinder01', 'vlinder02'])


#%%
# from datetime import datetime
# import pandas as pd

# startdt = datetime(2023, 3,24)
# enddt = datetime(2023, 3, 28)


era = dataset.get_modeldata(stations=['vlinder01'])

# era.get_ERA5_data(dataset.metadf, startdt, enddt)






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The class object for a Vlinder/mocca station
@author: thoverga
"""



plot_settings = {}
# =============================================================================
# General plot settings
# =============================================================================

# =============================================================================
# Default obs and metadata settings
# =============================================================================

#Static fields are fields (attributes and observations) that do not change in time
static_fields = ['network', 'name', 
                'lat', 'lon', #TODO make these dynamic, now used as static 
                'call_name', 'location',
                'lcz']

#Categorical fields are fields with values that are assumed to be categorical.
#Note: (there are static and dynamic fields that are categorical)
categorical_fields = ['wind_direction', 'lcz']


observation_types = ['temp', 'radiation_temp', 'humidity', 'precip',
                     'precip_sum', 'wind_speed', 'wind_gust', 'wind_direction',
                     'pressure', 'pressure_at_sea_level']

location_info = ['network', 'lat', 'lon', 'lcz', 'call_name', 'location' ]




# =============================================================================
# Timeseries plots
# =============================================================================
plot_settings['time_series'] = {
    
    #shape
    'figsize' : (10,5),
    'linewidth': 2, #
    'linezorder': 1, #for ok obs
    'scattersize': 4,
    'scatterzorder': 2
    
    }
# =============================================================================
# Spatial plot settings
# =============================================================================

plot_settings['spatial_geo'] = {
    #projection
    # 'proj' : 'Orthographic', #Orthographic or AlbersEqualArea
    'extent' : [ 2.260609, 49.25,  6.118359, 52.350618], #used if observatioons are within
    
    #colors
    # 'cmap' : 'Set1',
    'cmap' : 'inferno_r',
    'n_for_categorical' : 5, #number of quantiles for cat data (not for LCZ) 
    
    #shape
    'figsize': (10,15),
    
    #datetime
    'fmt': "%d/%m/%Y %H:%M:%S UTC" 
    }

# =============================================================================
# Stats plot settings
# =============================================================================

plot_settings['qc_stats'] = {
    #shape
    'figsize': (10,10),
    

    }

plot_settings['color_mapper']={
    #QC specific labels
    'duplicated_timestamp': '#a32a1f',
    'gross_value': '#f1ff2b',
    'persistance':'#f0051c',
    'repetitions':'#056ff0',
    'step':'#05d4f0',
    'window_variation':'#05f0c9',
    
    #missing and gap
    'gap': '#f00592',
    'missing_timestamp':'#e86bb6',
    
    #common
    'ok': '#07f72b',
    'not checked': '#f7cf07',
    
    #Aggregated
    'outlier': '#f20000'
    
    }




print_settings = {
    "fmt_datetime":"%d/%m/%Y %H:%M:%S",
    "max_print_per_line":"40"
    }


# =============================================================================
# variables display strings
# =============================================================================
vars_display = {
    'network': 'network',
    'name': 'station name',
    'call_name': 'pseudo name',
    'location': 'region',
    
    
    'lat': 'latitude',
    'lon': 'longtitude',
    
    'temp': 'temperature',
    'radiation_temp': 'radiation temperature',
    'humidity': 'humidity',
    'precip': 'precipitation intensity',
    'precip_sum': 'cummulated precipitation',
    'wind_speed': 'wind speed',
    'wind_gust': 'wind gust speed',
    'wind_direction': 'wind direction',
    'pressure': 'air pressure',
    'pressure_at_sea_level': 'corrected pressure at sea level',
    
    'lcz':'LCZ'
    
    }
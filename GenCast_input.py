import xarray as xr
import numpy as np
import dask
import cdsapi

import sys
import traceback  # traceback 모듈 추가

sys.path.append("/home/hiskim1/graphcast/lib")

import his_utils
import argparse

#==============================================================
#  Phase 0: Argument Parsing
#==============================================================
try:
    print("====================================================")
    print("                   Phase 0: Start                  ")
    print("====================================================")
    parser = argparse.ArgumentParser(description='Generate GenCast input data')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--day', type=str, required=True)
    parser.add_argument('--grid', type=str, required=False)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
except Exception as e:
    print("====================================================")
    print("                   Phase 0: Error                   ")
    print("An error occurred while parsing arguments")
    print("Exception traceback:")
    print(traceback.format_exc())  # 에러 발생 위치 전체 출력
    print("====================================================")
    sys.exit(1)

#==============================================================
#  Phase 1: Set Default Paths and Variables
#==============================================================
print("====================================================")
print("                   Phase 1: Setup                   ")
print("====================================================")

is_grib = True
default_path = "/geodata2/Gencast/tmp"

if is_grib:
    ds_surface = f"{default_path}/suf.grib"
    ds_surface_precip = f"{default_path}/tp.grib"
    ds_pressure_level = f"{default_path}/pres.grib"
    ds_surface_precip_prior = f"{default_path}/tp_prior.grib"
else:
    ds_surface = "testdata/surface.nc"
    ds_surface_precip = "testdata/precip.nc"
    ds_surface_precip_prior = None
    ds_pressure_level = "testdata/pressure.nc"

year = [f"{args.year}"]
month = [f"{args.month}"]
day = int(args.day)
time = ["00:00", "12:00"]

if not args.grid:
    print("No grid specified. Defaulting to 1.0/1.0")
    grid = "1.0/1.0"
else:
    grid = args.grid

if ds_surface_precip_prior is not None:
    if day == 1:  # If it's the first day of the month
        # Calculate previous month
        month_prior = args.month - 1
        year_prior = args.year
        
        if month_prior == 0:  # If it was January, go to previous year December
            month_prior = 12
            year_prior = args.year - 1
            
        # Get the last day of the previous month
        if month_prior in [4, 6, 9, 11]:
            prev_day = 30
        elif month_prior == 2:
            # Check for leap year
            if (year_prior % 4 == 0 and year_prior % 100 != 0) or (year_prior % 400 == 0):
                prev_day = 29
            else:
                prev_day = 28
        else:
            prev_day = 31
            
        day_prior = [f"{prev_day}"]
        month_prior = [f"{month_prior}"]
        year_prior = [f"{year_prior}"]
    else:
        day_prior = [f"{day-1}"]
        month_prior = month
        year_prior = year
        
    time_prior = [
        "12:00", "13:00", "14:00", "15:00", 
        "16:00", "17:00", "18:00", "19:00",
        "20:00", "21:00", "22:00", "23:00"
    ]

#==============================================================
#  Phase 2: Define CDS API Client and Requests
#==============================================================
print("====================================================")
print("      Phase 2: Define CDS API Client and Requests   ")
print("====================================================")

client = cdsapi.Client()
dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "data_format": "grib",
    "download_format": "unarchived"
}

P_level_13 = [
    "50", "100", "150", "200", "250", 
    "300", "400", "500", "600","700", 
    "850", "925", "1000"
]

dataset_single = "reanalysis-era5-single-levels"
dataset_multi = "reanalysis-era5-pressure-levels"

req_multi = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    "year": year,
    "month": month,
    "day": [f"{day}"],
    "time": time,
    "pressure_level": P_level_13,
    "data_format": "grib",
    "download_format": "unarchived",
    'grid': grid
}

req_tp = {
    "product_type": ["reanalysis"],
    "variable": ["total_precipitation"],
    "year": year,
    "month": month,
    "day": [f"{day}"],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    'grid': grid
}

req_surface = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "toa_incident_solar_radiation",
        "geopotential",
        "land_sea_mask"
    ],
    "year": year,
    "month": month,
    "day":[f"{day}"],
    "time": time,
    "data_format": "grib",
    "download_format": "unarchived",
    'grid': grid
}

#==============================================================
#  Phase 3: Retrieve Data via CDS API
#==============================================================
print("====================================================")
print("              Phase 3: Retrieving Data              ")
print("====================================================")

print("Retrieving total_precipitation (single-level) data...")
client.retrieve(
    dataset_single,
    req_tp,
    ds_surface_precip
)

print("Retrieving surface data (single-level)...")
client.retrieve(
    dataset_single,
    req_surface,
    ds_surface
)

print("Retrieving multi-level data (pressure levels)...")
client.retrieve(
    dataset_multi,
    req_multi,
    ds_pressure_level
)

if ds_surface_precip_prior is not None:
    print("Retrieving prior total_precipitation data...")
    req_tp_prior = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": year_prior,
        "month": month_prior,
        "day": day_prior,
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        'grid': '1.0/1.0'
    }
    client.retrieve(    
        dataset_single,
        req_tp_prior,
        ds_surface_precip_prior
    )

#==============================================================
#  Phase 4: Open and Pre-process Datasets
#==============================================================
print("====================================================")
print("          Phase 4: Open and Pre-process Data        ")
print("====================================================")

print("Opening ds_surface...")
if is_grib:
    ds1 = xr.open_dataset(ds_surface, engine='cfgrib')
    ds1 = ds1.drop_vars(['number', 'step', 'surface', 'valid_time'])
    ds1 = ds1.rename({"z" : "geopotential_at_surface"})
else:
    ds1 = xr.open_dataset(ds_surface)
    ds1 = ds1.rename({'var165':"u10",
                      'var166':"v10", 
                      'var167':"t2m", 
                      'var129':"geopotential_at_surface", 
                      'var172':"lsm", 
                      'var151':"msl", 
                      'var34': "sst",
                      'var212':"tisr"})

print("Opening and pre-processing ds_surface_precip (tp)...")
def sync_tp_coords(dataset: xr.Dataset):
    dataset = dataset.stack(new_time=['time', 'step'])
    dataset = dataset.assign_coords(new_time=dataset.valid_time.values)
    dataset = dataset.rename({'new_time': 'time'})
    dataset = dataset.drop_vars(['number', 'surface'])
    return dataset

if is_grib:
    ds2 = xr.open_dataset(ds_surface_precip, engine='cfgrib')
    if ds_surface_precip_prior is not None:
        prior = xr.open_dataset(ds_surface_precip_prior, engine='cfgrib')
        print("Syncing prior precipitation coords...")
        prior = sync_tp_coords(prior)

    print("Syncing ds2 coords...")
    ds2 = sync_tp_coords(ds2)

    print("Slicing ds2 to remove first 5 and last 7 timesteps...")
    ds2 = ds2.isel(time=slice(5, -7))
else:
    ds2 = xr.open_dataset(ds_surface_precip)
    ds2 = ds2.rename({'var228': 'tp'})
    if ds_surface_precip_prior is not None:
        prior = xr.open_dataset(ds_surface_precip_prior)
        prior = prior.rename({'var228': 'tp'})

# 2-2. the time before
if is_grib and ds_surface_precip_prior is not None:
    print("Slicing prior to keep indices [5:29]...")
    prior = prior.isel(time=slice(5, 29))

# 2-3. merge the two datasets
if ds_surface_precip_prior is not None:
    print("Concatenating prior and ds2 along time dimension...")
    ds2 = xr.concat([prior, ds2], dim='time')

print("Sorting ds2 by time...")
ds2 = ds2.sortby('time')

print("Resampling ds2 in 12h intervals and summing precipitation...")
ds2 = ds2.resample(time='12h', closed='right', label='right').sum()

print("Selecting final time slice from ds2 [2:4]...")
ds2 = ds2.isel(time=slice(2, 4))

print("Opening ds_pressure_level (ds3)...")
if is_grib:
    ds3 = xr.open_dataset(ds_pressure_level, engine='cfgrib')
    ds3 = ds3.drop_vars(['number', 'step', 'valid_time'])
    ds3 = ds3.rename({"isobaricInhPa" : "level"})
    ds3 = ds3.sortby('level', ascending=True)
else:
    ds3 = xr.open_dataset(ds_pressure_level)
    ds3 = ds3.rename({'var130':"t", 
                      'var131':"u", 
                      'var132':"v", 
                      'var129':"z", 
                      'var133':"q", 
                      'var135':"w",
                      'plev': "level"})
    ds3['level'] = ds3['level']/100
    ds3['level'].attrs['units'] = 'hPa'
    ds3['level'].attrs['long_name'] = 'pressure level'

level = ds3.level.values
level = level.astype(np.int32)
ds3 = ds3.assign_coords(level=('level', level))

#==============================================================
#  Phase 5: Merge All Datasets
#==============================================================
print("====================================================")
print("               Phase 5: Merging Datasets            ")
print("====================================================")

ds_list = [ds1, ds2, ds3]

print("Merging ds1, ds2, ds3...")
result = xr.merge(ds_list)

for ds_temp in ds_list:
    ds_temp.close()

print("Applying his_utils.transform_dataset(result, '12h')...")
result = his_utils.transform_dataset(result, "12h")

#==============================================================
#  Phase 6: Save Final NetCDF
#==============================================================
print("====================================================")
print("           Phase 6: Saving Final NetCDF             ")
print("====================================================")

result_path = args.output
print(f"Saving result to NetCDF file: {result_path}")
result.to_netcdf(result_path)

print("""
   Done! Your ERA5 data has been processed.
""")

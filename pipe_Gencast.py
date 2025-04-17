import dataclasses
import datetime
import math
from typing import Optional
import haiku as hk
import jax
import numpy as np
import xarray
import argparse
import sys
import traceback 
import pandas as pd
import numpy as np
import xarray as xr
from datetime import date, timedelta

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

print("""
 ███████╗████████╗ █████╗ ██████╗ ████████╗
 ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝
 ███████╗   ██║   ███████║██████╔╝   ██║   
 ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   
 ███████║   ██║   ██║  ██║██║  ██║   ██║   
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
""")

start_date = '2021-01-01'
end_date = '2021-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')


P_level_13 = [
    50, 100, 150, 200, 250, 
    300, 400, 500, 600, 700, 
    850, 925, 1000
]

PRESSURE_VARIABLES=[
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "vertical_velocity",
    "geopotential"
]

SURFACE_VARIABLES=[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "2m_temperature",
    "geopotential_at_surface",
    "land_sea_mask"
]


weatherbench = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")

wind = []

for date in date_range:

    date_before = date - timedelta(days=1)
    
    pres_surf = weatherbench[
        PRESSURE_VARIABLES + SURFACE_VARIABLES
        ].sel(
            level=P_level_13, 
            time=[f"{date}T00:00:00", f"{date}T12:00:00"]
            )

    tp = weatherbench[
        "total_precipitation"
        ].sel(
            time=slice(f"{date_before}T13:00:00", f"{date}T12:00:00")
        ).resample(time="12h", 
        closed="right", 
        label="right").sum()

    input = xr.merge(
        [pres_surf, tp]
        ).expand_dims(
            dim={'batch': [0]}
        ).rename({
            "latitude":"lat",
            "longitude":"lon"
        })

    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if 'batch' in input[var].dims:
            input[var] = input[var].squeeze('batch')

    
    #-----------------------------
    # Phase 0: Argument Parsing
    #-----------------------------
    try:
        parser = argparse.ArgumentParser(description='run GenCast. LEGGO')
        parser.add_argument('--model', 
                            type=str, 
                            choices=[
                                "0.25_2019", 
                                "0.25_operational_2022", 
                                "1.0_2019",
                                "1.0_mini_2019"
                                ], 
                            required=True)
        parser.add_argument('--eval_steps', type=int, required=True)
        parser.add_argument('--ens_num', type=int, required=True)
        parser.add_argument('--output', type=str, required=True)

        model_type = {
            "0.25_2019" : 'gencast_params_GenCast 0p25deg _2019.npz',
            "0.25_operational_2022" : 'gencast_params_GenCast 0p25deg Operational _2022.npz',
            "1.0_2019" : 'gencast_params_GenCast 1p0deg _2019.npz',
            "1.0_mini_2019" : 'gencast_params_GenCast 1p0deg Mini _2019.npz'
        }
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 0: Error                   ")
        sys.error.write("An error occurred while parsing arguments")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())  # 에러 발생 위치 전체 출력
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 1: Load the model
    #-----------------------------
    print("\n\n=======================================")
    print("        Phase 1: Load the model        ")
    print("=======================================")
    try:
        # FIXME: update the path of params
        MAIN_DIR= "/geodata2/S2S/DL/GC_input/"
        MODEL_PATH = MAIN_DIR + "params/" + model_type[parser.parse_args().model]
        with open(MODEL_PATH, "rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)
            denoiser_architecture_config = ckpt.denoiser_architecture_config
            # Following two lines required when the model runs on GPU
            denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
            denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
        params = ckpt.params
        state = {}

        task_config = ckpt.task_config
        sampler_config = ckpt.sampler_config
        noise_config = ckpt.noise_config
        noise_encoder_config = ckpt.noise_encoder_config
        denoiser_architecture_config = ckpt.denoiser_architecture_config
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 1: Error                   ")
        sys.error.write("An error occurred while loading the model")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())  # 에러 발생 위치 전체 출력
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 2: Load Input data
    #-----------------------------
    print("=======================================")
    print("       Phase 2: Load Input data        ")
    print("=======================================")
    try:
        input_data = input
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 2: Error                   ")
        sys.error.write("An error occurred while loading input data")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)

    
    #-----------------------------
    # Phase 3: Extract and Eval
    #-----------------------------
    print("\n\n=======================================")
    print("       Phase 3: Extract and Eval       ")
    print("=======================================")
    try:
        eval_inputs, _, _ = data_utils.extract_inputs_targets_forcings(
            input_data, 
            target_lead_times=slice("12h", "0h"),
            **dataclasses.asdict(task_config))
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 3: Error                   ")
        sys.error.write("An error occurred while extracting inputs and targets")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 4: Load Normalization data
    #-----------------------------
    print("\n\n=======================================")
    print("    Phase 4: Load Normalization data   ")
    print("=======================================")
    try:
        # FIXME: update the path of stat data
        STATS_DIR = MAIN_DIR + "stat/gencast_stats_"
        with open(STATS_DIR + "diffs_stddev_by_level.nc", "rb") as f:
            diffs_stddev_by_level = xarray.load_dataset(f).compute()
        with open(STATS_DIR + "mean_by_level.nc", "rb") as f:
            mean_by_level = xarray.load_dataset(f).compute()
        with open(STATS_DIR + "stddev_by_level.nc", "rb") as f:
            stddev_by_level = xarray.load_dataset(f).compute()
        with open(STATS_DIR + "min_by_level.nc", "rb") as f:
            min_by_level = xarray.load_dataset(f).compute()
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 4: Error                   ")
        sys.error.write("An error occurred while loading normalization data")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)



    #-----------------------------
    # Phase 5: Build jitted functions
    #-----------------------------
    print("\n\n=======================================")
    print("    Phase 5: Build jitted functions    ")
    print("=======================================")
    try:
        def construct_wrapped_gencast():
            """Constructs and wraps the GenCast Predictor."""
            predictor = gencast.GenCast(
                sampler_config=sampler_config,
                task_config=task_config,
                denoiser_architecture_config=denoiser_architecture_config,
                noise_config=noise_config,
                noise_encoder_config=noise_encoder_config,
            )

            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=diffs_stddev_by_level,
                mean_by_level=mean_by_level,
                stddev_by_level=stddev_by_level,
            )

            predictor = nan_cleaning.NaNCleaner(
                predictor=predictor,
                reintroduce_nans=True,
                fill_value=min_by_level,
                var_to_clean='sea_surface_temperature',
            )

            return predictor

        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            predictor = construct_wrapped_gencast()
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        @hk.transform_with_state
        def loss_fn(inputs, targets, forcings):
            predictor = construct_wrapped_gencast()
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
                (loss, diagnostics),
            )

        def grads_fn(params, state, inputs, targets, forcings):
            def _aux(params, state, i, t, f):
                (loss, diagnostics), next_state = loss_fn.apply(
                    params, state, jax.random.PRNGKey(0), i, t, f
                )
                return loss, (diagnostics, next_state)

            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                _aux, has_aux=True
            )(params, state, inputs, targets, forcings)
            return loss, diagnostics, next_state, grads

        loss_fn_jitted = jax.jit(
            lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
        )
        grads_fn_jitted = jax.jit(grads_fn)
        run_forward_jitted = jax.jit(
            lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
        )
        # We also produce a pmapped version for running in parallel.
        run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 5: Error                   ")
        sys.error.write("An error occurred while building jitted functions")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 6: Setup required frames and data
    #-----------------------------
    print("\n\n=======================================")
    print("Phase 6: Setup required frames and data")
    print("=======================================")
    try:
        eval_steps = parser.parse_args().eval_steps
        num_ensemble_members = parser.parse_args().ens_num

        rng = jax.random.PRNGKey(0)
        rngs = np.stack(
            [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0
        )
        chunks = []
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 6: Error                   ")
        sys.error.write("An error occurred while setting up frames and data")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)

    
    def create_forcing_dataset(time_steps, 
                            resolution, 
                            start_time,
                            d_t = 6):
        lon = np.arange(0.0, 360.0, resolution, dtype=np.float32)
        
        start_datetime = pd.to_datetime(start_time) + pd.Timedelta(hours=12)
        datetime = pd.date_range(start=start_datetime, 
                                periods=time_steps, 
                                freq=f"{d_t}h")
        
        
        time = pd.timedelta_range(start=pd.Timedelta(hours=d_t), 
                                periods=time_steps, 
                                freq=f"{d_t}h")
            
        variables = ['year_progress_sin',
                    'year_progress_cos',
                    'day_progress_sin',
                    'day_progress_cos']
        
        if d_t == 6:
            lat = np.arange(-90.0, 90.0 + resolution/2, resolution, dtype=np.float32)
            ds = xr.Dataset(
            coords={
                'lon': ('lon', lon),
                'lat': ('lat', lat),
                'datetime': ('datetime', datetime),
                'time': ('time', time)
                }
            )
            
            ds.lat.attrs['long_name'] = 'latitude'
            ds.lat.attrs['units'] = 'degrees_north'

            variables.append('toa_incident_solar_radiation')
            
            data_utils.add_tisr_var(ds)
        else:
            ds = xr.Dataset(
                    coords={
                        'lon': ('lon', lon),
                        'datetime': ('datetime', datetime),
                        'time': ('time', time)
                        }
                    )
        
        ds.lon.attrs['long_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees_east'

        data_utils.add_derived_vars(ds)
        
        # `datetime` is needed by add_derived_vars but breaks autoregressive rollouts.
        ds = ds.drop_vars("datetime")
        
        ds = ds[list(variables)]
        
        # 각 변수에 'batch' 차원 추가
        for var in variables:
            # 현재 변수의 차원과 데이터 가져오기
            current_dims = ds[var].dims
            current_data = ds[var].values

            # 'batch' 차원을 추가한 새로운 데이터 배열 생성
            new_shape = (1,) + current_data.shape
            perturbed = np.zeros(new_shape, dtype=current_data.dtype)
            perturbed[0] = current_data

            # 새로운 차원 순서 정의 ('batch'를 첫 번째로)
            new_dims = ('batch',) + current_dims

            # 새로운 DataArray 생성 및 할당 (coordinate는 추가하지 않음)
            ds[var] = xr.DataArray(
                data=perturbed,
                dims=new_dims,
                coords={dim: ds[dim] for dim in current_dims}  # 'batch'는 coordinate에 포함하지 않음
            )

        return ds



    def create_target_dataset(
            time_steps, 
            resolution, 
            pressure_levels, 
            d_t = 6):
        
        # Define coordinates
        lon = np.arange(0.0, 360.0, resolution, dtype=np.float32)
        lat = np.arange(-90.0, 90.0 + resolution/2, resolution, dtype=np.float32)
        
        if pressure_levels == 37:
            level = [   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125, 150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600, 650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975, 1000]
        elif pressure_levels == 13:
            level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        else:
            raise ValueError("Unsupported number of pressure levels. Choose either 37 or 13.")

        level = np.array(level, dtype=np.int64)

        # 시작 시간부터 time_steps 개의 6시간 간격 타임델타 생성
        time = pd.timedelta_range(start=pd.Timedelta(hours=d_t), 
                                periods=time_steps, 
                                freq=f"{d_t}h")
        # timedelta64[ns]로 명시적 변환
        # time = time.astype('timedelta64[ns]')/

        # Create the dataset
        ds = xr.Dataset(
            coords={
                'lon': ('lon', lon),
                'lat': ('lat', lat),
                'level': ('level', level.astype(np.int32)),
                'time': ('time', time),
            }
        )

        ds.lat.attrs['long_name'] = 'latitude'
        ds.lat.attrs['units'] = 'degrees_north'

        ds.lon.attrs['long_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees_east'

        surface_vars = ['2m_temperature', 
                            'mean_sea_level_pressure', 
                            '10m_v_component_of_wind', 
                            '10m_u_component_of_wind']
        # Add data variables filled with NaN
        if d_t == 6:
            surface_vars.append('total_precipitation_6hr')
        else:
            surface_vars.append('total_precipitation_12hr')
            surface_vars.append('sea_surface_temperature')
        
        level_vars = ['temperature', 'geopotential', 'u_component_of_wind', 
                    'v_component_of_wind', 'vertical_velocity', 'specific_humidity']

        for var in surface_vars:
            ds[var] = xr.DataArray(
                data=np.full((1, time_steps, len(lat), len(lon)), np.nan, dtype=np.float32),
                dims=['batch', 'time', 'lat', 'lon']
            )

        for var in level_vars:
            ds[var] = xr.DataArray(
                data=np.full((1, time_steps, len(level), len(lat), len(lon)), np.nan, dtype=np.float32),
                dims=['batch', 'time', 'level', 'lat', 'lon'],
            )

        ds = ds.transpose("batch", "time", "lat", "lon", ...)

        return ds


    # target_template 생성
    try:
        target_template = create_target_dataset(
            time_steps=int(eval_steps),
            resolution=float(parser.parse_args().model.split("_")[0]),
            pressure_levels=13,
            d_t=12
        )
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("               Phase 6 (target): Error              ")
        sys.error.write("An error occurred while generating target_template")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)

    # forcings 생성
    try:
        forcings = create_forcing_dataset(
            time_steps=int(eval_steps),
            resolution=float(parser.parse_args().model.split("_")[0]),
            start_time=input_data.datetime.values[0, 0],
            d_t=12
        )
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("              Phase 6 (forcings): Error             ")
        sys.error.write("An error occurred while generating forcings")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 7: Run the model
    #-----------------------------
    print("\n\n=======================================")
    print("          Phase 7: Run the model        ")
    print("=======================================")
    print(np.datetime64('now'))
    try:
        # The number of ensemble members should be a multiple of the number of devices.
        print("Note that the number of ensemble members should be a multiple of the number of devices.")
        print(f"Number of local devices {len(jax.local_devices())}")
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 7: Error                   ")
        sys.error.write("An error occurred while checking ensemble members and devices")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)

    try:
        for chunk in rollout.chunked_prediction_generator_multiple_runs(
            # Use pmapped version to parallelise across devices.
            predictor_fn=run_forward_pmap,
            rngs=rngs,
            inputs=eval_inputs,
            targets_template=target_template,
            forcings=forcings,
            num_steps_per_chunk=1,
            num_samples=num_ensemble_members,
            pmap_devices=jax.local_devices()
        ):
            chunks.append(chunk)
        predictions = xarray.combine_by_coords(chunks)
    except Exception as e:
        sys.error.write("====================================================")
        sys.error.write("                   Phase 7: Error                   ")
        sys.error.write("An error occurred during model prediction")
        sys.error.write("Exception traceback:")
        sys.error.write(traceback.format_exc())
        sys.error.write("====================================================")
        sys.exit(1)


    #-----------------------------
    # Phase 8: Save the output
    #-----------------------------
    print("=======================================")
    print("        Phase 8: Save the output       ")
    print("=======================================")
    print(np.datetime64('now'))

    predictions.squeeze().to_zarr(f"{date}.zarr")

    predictions[["u_component_of_wind", "v_component_of_wind"]].squeeze().to_zarr(f"uv_{date}.zarr")
    
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
import traceback  # traceback 모듈 추가

sys.path.append('/home/hiskim1/graphcast/lib')

import his_utils

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

'''
GenCast Rollout Code

arguments:
--model : model type (original, operational, small)

          Model list
          1. gencast_params_GenCast 0p25deg _2019.npz
          2. gencast_params_GenCast 0p25deg Operational _2022.npz
          3. gencast_params_GenCast 1p0deg _2019.npz
          4. gencast_params_GenCast 1p0deg Mini _2019.npz

--eval_steps : number of steps to evaluate
--ens_num : number of ensemble members

--input : input file path
--output : output file path

e.g.
python GenCast_run.py --model 0.25_2019 --eval_steps 5 --ens_num 10 --input /geodata2/S2S/DL/GC_input/data/ERA5_2019.nc --output /geodata2/S2S/DL/GC_output/GenCast_0.25_2019.nc
'''

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
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    model_type = {
        "0.25_2019" : 'gencast_params_GenCast 0p25deg _2019.npz',
        "0.25_operational_2022" : 'gencast_params_GenCast 0p25deg Operational _2022.npz',
        "1.0_2019" : 'gencast_params_GenCast 1p0deg _2019.npz',
        "1.0_mini_2019" : 'gencast_params_GenCast 1p0deg Mini _2019.npz'
    }
except Exception as e:
    print("====================================================")
    print("                   Phase 0: Error                   ")
    print("An error occurred while parsing arguments")
    print("Exception traceback:")
    print(traceback.format_exc())  # 에러 발생 위치 전체 출력
    print("====================================================")
    sys.exit(1)


#-----------------------------
# Phase 1: Load the model
#-----------------------------
print("\n\n=======================================")
print("        Phase 1: Load the model        ")
print("=======================================")
try:
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
    print("====================================================")
    print("                   Phase 1: Error                   ")
    print("An error occurred while loading the model")
    print("Exception traceback:")
    print(traceback.format_exc())  # 에러 발생 위치 전체 출력
    print("====================================================")
    sys.exit(1)


#-----------------------------
# Phase 2: Load Input data
#-----------------------------
print("=======================================")
print("       Phase 2: Load Input data        ")
print("=======================================")
try:
    DATA_PATH = parser.parse_args().input
    with open(DATA_PATH, "rb") as f:
        input_data = xarray.load_dataset(f).compute()
except Exception as e:
    print("====================================================")
    print("                   Phase 2: Error                   ")
    print("An error occurred while loading input data")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
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
    print("====================================================")
    print("                   Phase 3: Error                   ")
    print("An error occurred while extracting inputs and targets")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
    sys.exit(1)


#-----------------------------
# Phase 4: Load Normalization data
#-----------------------------
print("\n\n=======================================")
print("    Phase 4: Load Normalization data   ")
print("=======================================")
try:
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
    print("====================================================")
    print("                   Phase 4: Error                   ")
    print("An error occurred while loading normalization data")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
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
    print("====================================================")
    print("                   Phase 5: Error                   ")
    print("An error occurred while building jitted functions")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
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
    print("====================================================")
    print("                   Phase 6: Error                   ")
    print("An error occurred while setting up frames and data")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
    sys.exit(1)

# target_template 생성
try:
    target_template = his_utils.create_target_dataset(
        time_steps=int(eval_steps),
        resolution=float(parser.parse_args().model.split("_")[0]),
        pressure_levels=13,
        d_t=12
    )
except Exception as e:
    print("====================================================")
    print("               Phase 6 (target): Error              ")
    print("An error occurred while generating target_template")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
    sys.exit(1)

# forcings 생성
try:
    forcings = his_utils.create_forcing_dataset(
        time_steps=int(eval_steps),
        resolution=float(parser.parse_args().model.split("_")[0]),
        start_time=input_data.datetime.values[0, 0],
        d_t=12
    )
except Exception as e:
    print("====================================================")
    print("              Phase 6 (forcings): Error             ")
    print("An error occurred while generating forcings")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
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
    print("====================================================")
    print("                   Phase 7: Error                   ")
    print("An error occurred while checking ensemble members and devices")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
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
    print("====================================================")
    print("                   Phase 7: Error                   ")
    print("An error occurred during model prediction")
    print("Exception traceback:")
    print(traceback.format_exc())
    print("====================================================")
    sys.exit(1)


#-----------------------------
# Phase 8: Save the output
#-----------------------------
print("=======================================")
print("        Phase 8: Save the output       ")
print("=======================================")
print(np.datetime64('now'))

predictions.to_zarr(parser.parse_args().output)

print("""
 ███████╗███╗   ██╗██████╗ 
 ██╔════╝████╗  ██║██╔══██╗
 █████╗  ██╔██╗ ██║██║  ██║
 ██╔══╝  ██║╚██╗██║██║  ██║
 ███████╗██║ ╚████║██████╔╝
 ╚══════╝╚═╝  ╚═══╝╚═════╝ 
""")
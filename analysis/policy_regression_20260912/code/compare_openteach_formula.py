"""Offline comparison of historical/current Cartesian action arithmetic.

Only extracts pure computation methods with AST and constructs mock state
objects. Never instantiates a robot interface or connects to hardware.
Run using the openteach environment, which provides h5py and transform_utils.
"""

import ast
import glob
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
from deoxys.utils import transform_utils

REPO = Path('/home/jeremiah/openteach')
SOURCE = Path('openteach/components/operators/franka.py')
OUT = Path(__file__).resolve().parent
old = ast.parse(subprocess.check_output(
    ['git', 'show', f'f1ec57c:{SOURCE}'], cwd=REPO, text=True))
new = ast.parse((REPO / SOURCE).read_text())
old_fn = next(n for n in ast.walk(old) if isinstance(n, ast.FunctionDef) and n.name == 'arm_control')
new_fn = next(n for n in ast.walk(new) if isinstance(n, ast.FunctionDef) and n.name == 'get_abs_eef_pose_actions')
# Preserve initial state read and pose arithmetic; use the non-playback branch
# for the absolute target-pose policy. Exclude logging and actuator calls.
old_fn.body = old_fn.body[:3] + old_fn.body[3].orelse + [
    ast.Return(value=ast.Name(id='action', ctx=ast.Load()))]
old_fn.name = 'old_compute'
new_fn.name = 'new_compute'
module = ast.fix_missing_locations(ast.Module(body=[old_fn, new_fn], type_ignores=[]))
namespace = {'np': np, 'transform_utils': transform_utils,
             'TRANSLATION_VELOCITY_LIMIT': .1, 'ROTATION_VELOCITY_LIMIT': .2}
exec(compile(module, 'extracted_pure_methods', 'exec'), namespace)

paths = glob.glob('/data3/extracted_data/base_rollouts_original/*/deoxys*.h5')
paths += glob.glob('/data3/extracted_data/demonstration_dracorex_base_*/deoxys*.h5')
rows = []
for path in paths:
    with h5py.File(path) as handle:
        quat = handle['eef_quat'][:]
        pos = handle['eef_pos'][:]
        targets = handle['cartesian_pose_cmd'][:]
        actions = handle['arm_action'][:]
        if targets.ndim != 2 or targets.shape[1] != 7:
            continue
        formula_differences = []
        recorded_differences = []
        for i, target in enumerate(targets):
            mock = SimpleNamespace(robot_interface=SimpleNamespace(last_eef_quat_and_pos=(quat[i], pos[i])))
            old_action = np.asarray(namespace['old_compute'](mock, target, 0))
            current_action = np.asarray(namespace['new_compute'](mock, target))
            formula_differences.append(np.max(np.abs(old_action - current_action)))
            recorded_differences.append(np.max(np.abs(current_action - actions[i])))
        row = {
            'path': path,
            'n': len(targets),
            'old_current_formula_max_abs_error': float(np.max(formula_differences)),
            'recorded_formula_abs_error_quantile': np.percentile(recorded_differences, [50, 95, 99, 100]).tolist(),
        }
        rows.append(row)
        print(json.dumps(row))
(OUT / 'openteach_formula_comparison.json').write_text(json.dumps(rows, indent=2))

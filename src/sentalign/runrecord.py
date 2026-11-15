"""The JSON run record written next to the output of every file (``<name>.json``)."""
import json
import os
import platform
import subprocess

import sentalign


def _git_describe():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty', '--tags'], cwd=here, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def versions():
    import numpy
    import torch
    import transformers
    return {'python': platform.python_version(), 'numpy': numpy.__version__, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'sentalign': sentalign.__version__,
            'git_describe': _git_describe()}


def build_record(cfg, file_name, info, timings, total_seconds, n_links, n_pairs, input_paths=None, scoring=None, anchoring=None):
    settings = cfg.to_dict()
    return {
        'file': file_name,
        'input': input_paths,
        'settings': settings,
        'compat': cfg.compat,
        'model': cfg.model_name,
        'scoring': scoring,
        'deviations_from_paper_2023': cfg.deviations_from_paper(),
        'anchors': info.anchors,
        'anchoring': anchoring,
        'timings_seconds': dict(timings, total=total_seconds),
        # CPU seconds of the parent process as recorded by version 1 (the .info file)
        'timings_cpu_seconds': {'gale_church': info.gale_church_elapsed_time, 'calc_labse': info.calc_elapsed_labse,
                                'calc_anchors': info.calc_anchors_elapsed_time, 'greedy': info.greedy_algorithm_elapsed_time,
                                'align': info.align_elapsed_time},
        'counts': {'source_lines': info.source_file_length, 'target_lines': info.target_file_length,
                   'path_nodes': info.total_path_knots, 'links': n_links, 'aligned_pairs': n_pairs},
        'versions': versions(),
    }


def write_record(path, record):
    from sentalign.files import atomic_write
    atomic_write(path, json.dumps(record, indent=2, default=str) + '\n')

"""Short names for the encoders that were evaluated, so that ``--encoder bge-m3`` is enough.

A name here sets the encoder implementation and, unless given explicitly, the pooling.  The score cutoff is **not**
set: other models have other similarity ranges (see the README), so choose it for your data.
"""
# name: (encoder kind, model id, pooling or None)
MODEL_PRESETS = {
    'labse': ('hf', 'setu4993/LaBSE', 'pooler'),
    'bge-m3': ('hf', 'BAAI/bge-m3', 'cls'),
    'minilm': ('hf', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'mean'),
    'mpnet': ('hf', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'mean'),
    'laser2': ('laser', 'laser2', None),
    'laser3': ('laser', 'laser3', None),
}


def apply_preset(cfg):
    """Replace a short model name in ``cfg.model_name`` by the model and kind it stands for (in place)."""
    preset = MODEL_PRESETS.get(str(cfg.model_name).lower())
    if preset is None:
        return cfg
    kind, model, pooling = preset
    cfg.encoder_kind = kind
    cfg.model_name = model
    if pooling is not None and cfg.pooling == 'pooler':
        cfg.pooling = pooling
    return cfg

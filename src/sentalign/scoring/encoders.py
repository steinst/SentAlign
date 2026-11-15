"""Sentence encoders: texts -> L2-normalised float32 vectors.

``HFEncoder`` wraps any Hugging Face encoder (LaBSE, a fine-tuned LaBSE, other BERT-like models) and, with
``order='file'`` and ``batch_size=128``, encodes exactly as SentAlign 1.0 did.
"""
import hashlib
import json
import logging
import os
import re

import numpy as np

log = logging.getLogger('sentalign.encoders')

POOLINGS = ('pooler', 'cls', 'mean')
ORDERS = ('file', 'length')


class ModelLoadError(RuntimeError):
    """The model or tokenizer could not be loaded or moved to the device."""


class EncodingError(RuntimeError):
    """Encoding failed even with batch size 1 (e.g. CUDA out of memory)."""


def _is_oom(exc):
    import torch
    if isinstance(exc, getattr(torch, 'OutOfMemoryError', ())):
        return True
    if isinstance(exc, getattr(torch.cuda, 'OutOfMemoryError', ())):
        return True
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cudaerrormemoryallocation' in msg


def prepare_text(text):
    """Text as version 1 handed it to the tokenizer: stripped, runs of '_' collapsed (``create_list``)."""
    return re.sub('[_]+', '_', text.strip())


class HFEncoder:
    """Encoder for a Hugging Face model id or a local model folder."""

    kind = 'hf'

    def __init__(self, model_name='setu4993/LaBSE', device='cpu', pooling='pooler', batch_size=128, order='file',
                 max_tokens=512, revision=None, dedupe=False, half=False, trust_remote_code=False):
        if pooling not in POOLINGS:
            raise ValueError('pooling must be one of %s' % (POOLINGS,))
        if order not in ORDERS:
            raise ValueError('order must be one of %s' % (ORDERS,))
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.batch_size = int(batch_size)
        self.order = order
        self.max_tokens = int(max_tokens)
        self.revision = revision
        self.dedupe = bool(dedupe)
        self.trust_remote_code = bool(trust_remote_code)   # run the model code that comes with the model
        self.half = bool(half)        # run the model in float16 (GPU only); vectors are returned as float32
        self.tokenizer = None
        self.model = None
        self.oom_retries = 0

    # ------------------------------------------------------------------ model
    def load(self):
        if self.model is not None:
            return
        if self.half and not str(self.device).startswith('cuda'):
            raise ModelLoadError('half precision needs a CUDA device (got %r)' % (self.device,))
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        log.info('loading encoder %s on %s', self.model_name, self.device)
        try:
            from transformers import AutoModel, AutoTokenizer
            kw = {'revision': self.revision} if self.revision else {}
            if self.trust_remote_code:
                kw['trust_remote_code'] = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kw)
            model = AutoModel.from_pretrained(self.model_name, **kw)
            model = model.eval().to(self.device)
            if self.half:
                model = model.half()
            self.model = model
        except Exception as e:  # any loading or device error is reported as one clear error
            hint = ''
            if not self.trust_remote_code and 'trust_remote_code' in str(e):
                hint = (' (this model comes with its own code; --trust-remote-code runs it, which you should only '
                        'allow for a model you trust)')
            raise ModelLoadError('could not load %s on device %r: %s%s' % (self.model_name, self.device, e, hint)) from e

    def describe(self):
        d = {'kind': self.kind, 'model': self.model_name, 'revision': self.revision, 'pooling': self.pooling,
             'max_tokens': self.max_tokens, 'order': self.order, 'batch_size': self.batch_size,
             'dedupe': self.dedupe, 'half': self.half, 'device': str(self.device)}
        if self.model is not None:
            d['model_class'] = type(self.model).__name__
            d['resolved_revision'] = getattr(getattr(self.model, 'config', None), '_commit_hash', None)
        return d

    def fingerprint(self):
        """Everything that can change the vectors (used as the cache key)."""
        d = self.describe()
        d.pop('dedupe', None)
        if not d.get('half'):
            d.pop('half', None)             # float32 caches made before the option existed stay valid
        d['device'] = str(self.device).split(':')[0]
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:20]

    # ------------------------------------------------------------------ encoding
    def _pool(self, outputs, attention_mask):
        import torch
        if self.pooling == 'pooler':
            return outputs.pooler_output
        hidden = outputs.last_hidden_state
        if self.pooling == 'cls':
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def _encode_batch(self, batch):
        import torch
        import torch.nn.functional as F
        inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=self.max_tokens,
                                add_special_tokens=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            vectors = F.normalize(self._pool(outputs, inputs['attention_mask']).float(), p=2)
            return vectors.cpu().detach().numpy()

    def _batches(self, n_texts, lengths):
        if self.order == 'file':
            order = np.arange(n_texts)
        else:
            order = np.argsort(-np.asarray(lengths), kind='stable')
        return order

    def encode(self, texts):
        """(len(texts), d) float32, rows L2-normalised."""
        self.load()
        texts = [prepare_text(t) for t in texts]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        unique_index = None
        if self.dedupe:
            first = {}
            unique_index = np.asarray([first.setdefault(t, len(first)) for t in texts], dtype=np.int64)
            work = list(first)
        else:
            work = texts
        order = self._batches(len(work), [len(t) for t in work])
        out = None
        pos = 0
        batch_size = self.batch_size
        while pos < len(order):
            idx = order[pos:pos + batch_size]
            try:
                vec = self._encode_batch([work[i] for i in idx])
            except Exception as e:
                if not _is_oom(e):
                    raise
                self._empty_cache()
                if batch_size == 1:
                    raise EncodingError('out of memory while encoding one text of %d characters on %s'
                                        % (len(work[idx[0]]), self.device)) from e
                batch_size = max(1, batch_size // 2)
                self.oom_retries += 1
                log.warning('out of memory on %s: encoding again with batch size %d%s', self.device, batch_size,
                            ' (vectors may differ from version 1 in the last digits)' if self.order == 'file' else '')
                continue
            if out is None:
                out = np.empty((len(work), vec.shape[1]), dtype=np.float32)
            out[idx] = vec
            pos += len(idx)
        if unique_index is not None:
            out = out[unique_index]
        return out

    def _empty_cache(self):
        if str(self.device).startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers for the legacy code
    def encode_one_batch(self, texts):
        """All texts in one forward pass, unnormalised pooler output as a torch tensor (version-1
        ``score_labse_matrix``; used only by the paper-2023 anchoring)."""
        import torch
        self.load()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512,
                                add_special_tokens=True).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).pooler_output

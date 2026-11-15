"""Optional on-disk cache of span vectors: ``<root>/<encoder fingerprint>/<sha256 of the texts>.npy``.

Useful when the same documents are aligned several times with different path-search or anchoring settings.
"""
import hashlib
import logging
import os
import tempfile

import numpy as np

log = logging.getLogger('sentalign.cache')


def texts_key(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, root):
        self.root = os.path.abspath(root)

    def _path(self, fingerprint, texts):
        return os.path.join(self.root, fingerprint, texts_key(texts) + '.npy')

    def load(self, fingerprint, texts):
        path = self._path(fingerprint, texts)
        if not os.path.exists(path):
            return None
        try:
            arr = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as e:
            log.warning('ignoring unreadable cache file %s (%s)', path, e)
            return None
        if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.dtype != np.float32:
            log.warning('ignoring cache file %s with shape %s', path, arr.shape)
            return None
        log.debug('cache hit %s', path)
        return arr

    def save(self, fingerprint, texts, vectors):
        path = self._path(fingerprint, texts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                np.save(f, np.ascontiguousarray(vectors, dtype=np.float32), allow_pickle=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

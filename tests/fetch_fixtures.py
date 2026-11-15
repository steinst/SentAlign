"""Download the large test fixtures (not in git) and extract them to tests/fixtures/eval/.

    python tests/fetch_fixtures.py                  # download from the url in tests/fixtures_manifest.json
    python tests/fetch_fixtures.py --from-dir DIR   # use a local copy of the archive(s) (matched by file name)

The archive is verified against the sha256 and size of the manifest before it is extracted.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, 'fixtures_manifest.json')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive, target):
    """Extract a tar.gz, refusing absolute paths, '..' and links that leave the target folder."""
    target = os.path.realpath(target)
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar.getmembers():
            dest = os.path.realpath(os.path.join(target, m.name))
            if not (dest == target or dest.startswith(target + os.sep)):
                raise ValueError('unsafe path in archive: ' + m.name)
            if m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise ValueError('unexpected member type in archive: ' + m.name)
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(target, filter='data')
        else:
            tar.extractall(target)


def fetch(entry, from_dir=None, force=False):
    target = os.path.join(ROOT, entry['extract_to'])
    marker = os.path.join(target, '.fixtures-sha256')
    if not force and os.path.exists(marker) and open(marker).read().strip() == entry['sha256']:
        print('%s: up to date in %s' % (entry['name'], target))
        return
    with tempfile.TemporaryDirectory(prefix='sentalign_fixtures_') as tmp:
        archive = os.path.join(tmp, entry['name'])
        if from_dir:
            src = os.path.join(from_dir, entry['name'])
            if not os.path.isfile(src):
                raise SystemExit('not found: ' + src)
            shutil.copyfile(src, archive)
        else:
            url = entry['url']
            if not url.startswith(('http://', 'https://')):
                raise SystemExit('the manifest has no download url yet (%r); use --from-dir DIR' % url)
            print('downloading', url)
            urllib.request.urlretrieve(url, archive)
        size = os.path.getsize(archive)
        if size != entry['size']:
            raise SystemExit('%s: size %d, expected %d' % (entry['name'], size, entry['size']))
        digest = sha256_of(archive)
        if digest != entry['sha256']:
            raise SystemExit('%s: sha256 %s, expected %s' % (entry['name'], digest, entry['sha256']))
        os.makedirs(target, exist_ok=True)
        safe_extract(archive, target)
        with open(marker, 'w') as f:
            f.write(entry['sha256'] + '\n')
    print('%s: extracted to %s' % (entry['name'], target))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--from-dir', metavar='DIR', help='folder with a local copy of the archive(s)')
    parser.add_argument('--force', action='store_true', help='extract again even if the fixtures are up to date')
    args = parser.parse_args(argv)
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    for entry in manifest['archives']:
        fetch(entry, args.from_dir, args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""The README and CITATION.cff agree with the code: every option in the option tables, every path-finder setting,
every publication of citations.py with its BibTeX, the version."""
import os
import re

import pytest

import sentalign
from sentalign import cli
from sentalign.citations import ENTRIES
from sentalign.pathfinding.chain_finder import ChainFinder
from sentalign.pathfinding.vecalign_finder import VecalignFinder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return f.read()


def _table_cells():
    """The first two cells (long and short name) of every table row of the README, without the backticks."""
    cells = set()
    for line in _read('README.md').splitlines():
        if line.startswith('|') and not line.startswith('|---'):
            for cell in line.strip('|').split('|')[:2]:
                cells.update(part.strip().strip('`') for part in cell.split(' / '))
    return cells


def test_every_option_is_in_a_table():
    cells = _table_cells()
    missing = []
    for action in cli.build_parser()._actions:
        names = list(action.option_strings) or ['SRC_FILE TRG_FILE']
        missing += [n for n in names if n not in cells]
    assert not missing, missing
    assert '--no-half' in cells


@pytest.mark.parametrize('finder', [VecalignFinder, ChainFinder])
def test_every_pathfinder_setting_is_in_a_table(finder):
    cells = _table_cells()
    missing = [k for k in finder.option_defaults if k not in cells]
    assert not missing, missing


def test_every_publication_is_in_the_readme():
    text = _read('README.md')
    for key, entry in ENTRIES.items():
        if key == 'sentalign2':                        # a placeholder until the paper is published
            continue
        assert entry['bibtex'] in text, key


def test_citation_cff_version_and_paper():
    cff = _read('CITATION.cff')
    assert re.search(r'^version: (\S+)$', cff, re.M).group(1) == sentalign.__version__
    doi = re.search(r'doi = "([^"]+)"', ENTRIES['sentalign']['bibtex']).group(1)
    assert 'doi: %s' % doi in cff


def test_readme_has_no_placeholders_left_but_the_date():
    text = _read('README.md')
    assert 'Released 2026-MM-DD' in text or re.search(r'Released \d{4}-\d\d-\d\d', text)
    assert not re.search(r'\b[A-Z]+_[A-Z0-9_]+\b', text.replace('SRC_FILE', '').replace('TRG_FILE', '')
                         .replace('OUT_DIR', ''))


def test_version_is_the_same_everywhere():
    version = sentalign.__version__
    assert re.search(r'^version = "([^"]+)"$', _read('rust/pyproject.toml'), re.M).group(1) == version
    assert re.search(r'^__version__ = "([^"]+)"$', _read('rust/python/sentalign_rust/__init__.py'), re.M).group(1) == version
    # Cargo writes a pre-release as 2.0.0-rc.1 where Python has 2.0.0rc1
    cargo = re.search(r'^version = "([^"]+)"$', _read('rust/Cargo.toml'), re.M).group(1)
    assert re.sub(r'-([a-z]+)\.', r'\1', cargo) == version
    assert re.search(r'name = "sentalign-rust"\nversion = "%s"' % re.escape(cargo), _read('rust/Cargo.lock'))


def test_release_date_is_the_same_everywhere():
    """A released version has one date in the README, CITATION.cff and its CHANGELOG heading."""
    version = sentalign.__version__
    readme = re.search(r'^Released (\S+)\.$', _read('README.md'), re.M).group(1)
    if re.search(r'(a|b|rc|dev)\d+$', version):
        return                                      # a pre-release has no date yet
    cff = re.search(r'^date-released: (\S+)$', _read('CITATION.cff'), re.M)
    changelog = re.search(r'^## %s \((\S+)\)$' % re.escape(version), _read('CHANGELOG.md'), re.M)
    assert re.fullmatch(r'\d{4}-\d\d-\d\d', readme)
    assert cff and changelog and cff.group(1) == readme == changelog.group(1)


def test_readme_links_are_absolute():
    """PyPI shows the README without the repository, so a relative link there leads nowhere."""
    assert re.findall(r'\]\((?!https?://|#)([^)]*)\)', _read('README.md')) == []

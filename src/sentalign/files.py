"""Which files are aligned, which are done, atomic output; and the ``sentalign-files`` command.

``sentalign-files`` keeps the behaviour of the old ``files2align.py`` (it writes ``files2align.txt`` and
``filesdone.txt``).  The main ``sentalign`` command no longer needs those lists.
"""
import argparse
import os
import sys


def list_source_files(source_folder):
    """All regular files of the source folder, in sorted order."""
    return sorted(f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)))


def is_done(output_folder, name):
    """A file is done when both ``<out>/<name>.path`` and ``<out>/<name>.aligned`` exist."""
    return (os.path.exists(os.path.join(output_folder, name + '.path')) and
            os.path.exists(os.path.join(output_folder, name + '.aligned')))


def select_files(names, output_folder, overwrite=False):
    """Split ``names`` (processed in sorted order) into (to_align, already_done)."""
    to_align, done = [], []
    for name in sorted(set(names)):
        if not overwrite and is_done(output_folder, name):
            done.append(name)
        else:
            to_align.append(name)
    return to_align, done


def atomic_write(path, text):
    """Write to ``path + '.tmp'`` and rename when complete."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fo:
            fo.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    """Generate files2align.txt and filesdone.txt in the corpus folder (the old files2align.py)."""
    parser = argparse.ArgumentParser(prog='sentalign-files', description=__doc__.split('\n')[0])
    parser.add_argument('--corpus-folder', '-dir', required=True)
    parser.add_argument('--source-language', '-sl', default='eng')
    parser.add_argument('--output-folder', '-out', default='output')
    args = parser.parse_args(argv)

    corpus_folder = args.corpus_folder
    output_folder = os.path.join(corpus_folder, args.output_folder)
    source_language_folder = os.path.join(corpus_folder, args.source_language)
    if not os.path.isdir(source_language_folder):
        parser.error('source language folder not found: ' + source_language_folder)
    os.makedirs(output_folder, exist_ok=True)

    files_to_process = list_source_files(source_language_folder)
    files_not_to_process = sorted(f.replace('.aligned', '').replace('overlaps.', '') for f in os.listdir(output_folder)
                                  if f.endswith('.aligned'))

    with open(os.path.join(corpus_folder, 'files2align.txt'), 'w') as f:
        for file in files_to_process:
            f.write(file + '\n')
    with open(os.path.join(corpus_folder, 'filesdone.txt'), 'w') as f:
        for file in files_not_to_process:
            f.write(file + '\n')

    print('files2align.txt and filesdone.txt created in ' + corpus_folder + '/')
    print('Files to align: ' + str(len(files_to_process)))
    print('Files already aligned: ' + str(len(files_not_to_process)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

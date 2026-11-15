"""Configuration of a SentAlign run: one dataclass holding every option, with the 2023 defaults."""
import argparse
import dataclasses
from dataclasses import dataclass
from typing import List, Optional

COMPAT_PRESETS = ("paper-2023", "none")

# Settings that a preset decides unless they are given explicitly (None in the dataclass = "from the preset").
# paper-2023: everything as SentAlign 1.0.  none: the version-2 behaviour (the encoder, scoring, anchoring and path search differ).
PRESET_VALUES = {
    'paper-2023': {'batch_order': 'file', 'dedupe_texts': False, 'anchoring_delimiter': 4000000, 'batch_texts': 0},
    'none': {'batch_order': 'length', 'dedupe_texts': True, 'anchoring_delimiter': 100000000, 'batch_texts': 200000},
}
# link rewards of the sentalign-dp path search and their default score cutoff
REWARDS = {'sentalign': 0.4, 'squared-cosine': 0.3}
# what a run must keep to give the SentAlign 1.0 results
PAPER_2023_FIXED = {'pooling': 'pooler', 'batch_size': 128, 'batch_order': 'file', 'dedupe_texts': False, 'half': False,
                    'max_tokens': 512, 'model_name': 'setu4993/LaBSE', 'scorer': 'embedding', 'encoder_kind': 'hf'}


@dataclass
class AlignConfig:
    # input and output locations
    corpus_folder: Optional[str] = None
    source_language: str = 'eng'
    target_language: str = 'isl'
    filename: Optional[List[str]] = None
    output_folder: str = 'output'          # relative to the corpus folder (an absolute path is used as it is)
    # aligner settings (same names and defaults as the 2023 command line)
    max_concatenations: int = 4
    free_concatenations: int = 2
    score_cutoff: Optional[float] = None   # None = 0.4 with the SentAlign reward, 0.3 with --reward squared-cosine
    minimum_length_words: int = 1
    maximum_length_words: int = 110
    penalty_after_words: int = 80
    penalty_per_word: float = 0.01
    anchoring_delimiter: Optional[int] = None   # node limit before anchoring; None = from the preset
    maximum_length_gale_church: float = 10000
    # other settings
    proc_device: str = 'auto'              # auto = cuda when PyTorch finds a CUDA device, else cpu (see resolve_device)
    num_proc: int = 8
    # behaviour preset: none (version 2, the default) or paper-2023 (the results of SentAlign 1.0)
    compat: str = 'none'
    overwrite: bool = False
    # constants that were module-level globals in sentAlign.py (not on the command line)
    model_name: str = 'setu4993/LaBSE'
    max_length: int = 128                  # passed to create_emb_file (not used for truncation; kept as in 1.0)
    batch_size: int = 128
    cutoff_penalty: float = 1.25
    minimum_labse_anchor: float = 0.95
    labse_subtraction: float = 0.05
    abs_minimum_labse_anchor: float = 0.8
    # where the old .info text file goes; None means <corpus folder>/align_info/
    info_folder: Optional[str] = None
    # scoring
    scorer: str = 'embedding'
    encoder_kind: str = 'hf'
    pooling: str = 'pooler'                # pooler | cls | mean
    max_tokens: int = 512
    model_revision: Optional[str] = None
    batch_order: Optional[str] = None      # file | length; None = from the preset
    dedupe_texts: Optional[bool] = None    # encode identical texts once; None = from the preset
    embedding_cache: Optional[str] = None  # folder of the optional embedding cache
    trust_remote_code: bool = False        # allow an encoder that ships its own model code (it is executed)
    half: Optional[bool] = None            # encode with the model in float16; None = on for a CUDA device with
                                           # --compat none, off otherwise (float16 is not the paper's numbers).  With
                                           # proc_device auto it stays None until the device is known
    batch_texts: Optional[int] = None      # encode small files together, up to this many texts at a time (0 = one file
                                           # at a time); None = from the preset
    # anchoring
    anchors: str = 'auto'                  # auto | none | mutual-best | gale-church | legacy-greedy | file:PATH
    anchor_margin: float = 0.05            # an anchor must beat the second-best match by this much, on both sides
    anchor_min_words: int = 4              # ... and both sentences must have at least this many words
    max_segment_nodes: int = 4000000000    # refuse to solve a segment above this (one byte per node)
    # path search
    pathfinder: str = 'sentalign-dp'
    path_backend: str = 'auto'             # auto | cython | rust | python | legacy (the 2023 code; paper-2023 only)
    reward: str = 'sentalign'              # sentalign | squared-cosine (the link weight of ChainAlign)
    length_prior: bool = False             # squared-cosine only: multiply the link reward by the length-ratio prior
    pathfinder_options: Optional[dict] = None   # the path finder's own settings (--pathfinder-opt NAME=VALUE)

    def __post_init__(self):
        self.resolve()

    def resolve(self):
        """Fill the preset-dependent settings that were not given."""
        if self.compat not in PRESET_VALUES:
            raise ValueError('unknown --compat preset %r (known: %s)' % (self.compat, ', '.join(COMPAT_PRESETS)))
        if self.pathfinder != 'sentalign-dp' and self.compat == 'paper-2023':
            raise ValueError('--pathfinder %s needs --compat none (paper-2023 is the SentAlign 2023 path search)'
                             % self.pathfinder)
        if self.scorer != 'embedding' and self.compat == 'paper-2023':
            raise ValueError('--scorer %s needs --compat none (paper-2023 scores with LaBSE embeddings)' % self.scorer)
        if self.path_backend == 'legacy' and self.compat != 'paper-2023':
            raise ValueError('--path-backend legacy is the 2023 path search and needs --compat paper-2023')
        if self.max_concatenations > 15 and self.path_backend != 'legacy':
            # the kernels keep a link as one uint8 back-pointer, 3 + (ks-1)*K + (kt-1), which fits for K <= 15 only
            # version 1 had no such limit: its own path search (slow) still runs it
            hint = ' (the 2023 path search has no limit: --path-backend legacy)' if self.compat == 'paper-2023' else ''
            raise ValueError('--max-concatenations %d is too large: the path search supports at most 15%s'
                             % (self.max_concatenations, hint))
        for k, v in PRESET_VALUES[self.compat].items():
            if getattr(self, k) is None:
                setattr(self, k, v)
        if self.reward not in REWARDS:
            raise ValueError('unknown --reward %r (known: %s)' % (self.reward, ', '.join(REWARDS)))
        if self.reward != 'sentalign' and self.compat == 'paper-2023':
            raise ValueError('--reward %s needs --compat none (paper-2023 is the SentAlign reward)' % self.reward)
        if self.length_prior and self.reward != 'squared-cosine':
            raise ValueError('--length-prior belongs to --reward squared-cosine')
        if self.score_cutoff is None:
            self.score_cutoff = REWARDS[self.reward]
        from sentalign.scoring.models import apply_preset
        apply_preset(self)             # a short model name (labse, bge-m3, minilm, mpnet, laser2, laser3)
        if self.half is None:
            if self.compat != 'none' or self.encoder_kind != 'hf':
                self.half = False
            elif self.proc_device != 'auto':   # auto: decided with the device, in resolve_device
                self.half = str(self.proc_device).startswith('cuda')
        return self

    def resolve_device(self):
        """Settle the device: ``auto`` becomes cuda when PyTorch finds a CUDA device and cpu otherwise, and the
        default of ``half`` follows the device that is used.  A CUDA device that was asked for by name and is not
        there is an error (a run that was meant for the GPU should not take hours on the CPU without notice).
        Imports torch, so the aligner calls it when it is built, not ``resolve``."""
        if self.proc_device == 'auto':
            self.proc_device = default_device()
        elif str(self.proc_device).startswith('cuda') and not _cuda_available():
            raise ValueError('--proc-device %s: PyTorch finds no CUDA device here; leave the option out (the CPU is '
                             'then used) or give --proc-device cpu' % self.proc_device)
        if self.half is None:
            self.half = str(self.proc_device).startswith('cuda')
        return self

    def deviations_from_paper(self):
        """Settings with which a paper-2023 run no longer gives the SentAlign 1.0 results."""
        if self.compat != 'paper-2023':
            return {}
        return {k: getattr(self, k) for k, v in PAPER_2023_FIXED.items() if getattr(self, k) != v}

    def to_dict(self):
        return dataclasses.asdict(self)


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def default_device():
    """The device of ``--proc-device auto``: cuda when PyTorch finds a CUDA device, else cpu."""
    return 'cuda' if _cuda_available() else 'cpu'


class _KeyValue(argparse.Action):
    """NAME=VALUE ... -> dict (values stay text; the receiver converts them)."""

    def __call__(self, parser, namespace, values, option_string=None):
        out = dict(getattr(namespace, self.dest, None) or {})
        for item in values:
            if '=' not in item:
                parser.error('%s expects NAME=VALUE, got %r' % (option_string, item))
            key, value = item.split('=', 1)
            out[key.replace('-', '_')] = value
        setattr(namespace, self.dest, out)


def add_alignment_arguments(parser: argparse.ArgumentParser):
    """The command line flags of version 1, unchanged (names, types, defaults)."""
    d = AlignConfig()
    # Input and output file locations
    parser.add_argument('--corpus-folder', '-dir', help='Corpus folder with one sub-folder per language')
    # no parser default: the command line must be able to tell "not given" from an explicit eng / isl (pair form)
    parser.add_argument('--source-language', '-sl', default=None,
                        help='source language: its sub-folder of the corpus folder, and the language told to an '
                             'encoder that needs it (default with -dir: %s)' % d.source_language)
    parser.add_argument('--target-language', '-tl', default=None,
                        help='target language (default with -dir: %s)' % d.target_language)
    parser.add_argument('--filename', '-f', help='Name of source and target file(s) to be aligned', type=str, nargs='+')
    parser.add_argument('--output-folder', '-out', default=d.output_folder,
                        help='Output folder, relative to the corpus folder (default: %(default)s)')
    # Aligner settings
    parser.add_argument('--max-concatenations', '-concats', type=int, help='Maximum number of concatenated sentences per language (at most 15)', default=d.max_concatenations)
    parser.add_argument('--free-concatenations', '-freejoins', type=int, help='Maximum number of concatenations before penalty is applied', default=d.free_concatenations)
    parser.add_argument('--score-cutoff', '-cutoff', type=float, help='Minimum similarity score for a sentence pair to be considered (default: 0.4; 0.3 with --reward squared-cosine)', default=None)
    parser.add_argument('--minimum-length-words', '-minwords', type=int, help='Minimum number of words per language, for a sentence pair to be considered', default=d.minimum_length_words)
    parser.add_argument('--maximum-length-words', '-maxwords', type=int, help='Maximum number of words per language, for a sentence pair to be considered', default=d.maximum_length_words)
    parser.add_argument('--penalty-after-words', '-penwords', type=int, help='Maximum number of words per language, before a length penalty is applied', default=d.penalty_after_words)
    parser.add_argument('--penalty-per-word', '-wordpen', type=float, help='Penalty applied for each word when maximum number of unpenalized words have been reached', default=d.penalty_per_word)
    parser.add_argument('--anchoring-delimiter', '-anchor', type=int, default=None,
                        help='Maximum nodes in the alignment graph before the document is cut at anchors '
                             '(default: 100000000; 4000000 with --compat paper-2023)')
    parser.add_argument('--maximum-length-gale-church', '-maxgc', type=float, help='Maximum number of sentences in file for Gale-Church alignment. If longer, only greedy alignment selection applied', default=d.maximum_length_gale_church)
    # Other settings
    parser.add_argument('--proc-device', '-device', default=d.proc_device,
                        help='PyTorch device for the encoder: auto (default: cuda when an NVIDIA GPU with CUDA is found, '
                             'else cpu), cuda, cuda:1, cpu. cuda without such a GPU is an error')
    parser.add_argument('--num-proc', '-proc', type=int, help='number of processors to allocate for the pathfinding calculations', default=d.num_proc)
    # Scoring (version 2)
    g = parser.add_argument_group('scoring')
    g.add_argument('--scorer', default=d.scorer,
                   help='scorer (default: %(default)s; more can be added through the entry-point group sentalign.scorers)')
    g.add_argument('--encoder', dest='model_name', default=d.model_name,
                   help='sentence encoder: a short name (labse, bge-m3, minilm, mpnet, laser2, laser3), a Hugging Face model '
                        'id or a local folder, e.g. a fine-tuned LaBSE (default: %(default)s). Other models need their own '
                        '--score-cutoff')
    g.add_argument('--encoder-kind', default=d.encoder_kind,
                   help='encoder implementation (default: %(default)s; more through the entry-point group sentalign.encoders)')
    g.add_argument('--pooling', choices=('pooler', 'cls', 'mean'), default=d.pooling,
                   help='sentence vector from the model: LaBSE uses pooler, most sentence-transformers models mean (default: %(default)s)')
    g.add_argument('--model-revision', default=None, help='model revision on the Hugging Face hub (tag, branch or commit)')
    g.add_argument('--max-tokens', type=int, default=d.max_tokens, help='truncation length in tokens (default: %(default)s)')
    g.add_argument('--encode-batch-size', dest='batch_size', type=int, default=d.batch_size,
                   help='texts per encoder batch, halved automatically on out-of-memory (default: %(default)s)')
    g.add_argument('--batch-order', choices=('file', 'length'), default=None,
                   help='file: batches in file order as version 1 (paper-2023); length: sorted by length, faster (--compat none)')
    g3 = parser.add_argument_group('anchoring (documents above the node limit)')
    g3.add_argument('--anchors', default=d.anchors, metavar='STRATEGY',
                    help='auto (default: mutual-best; with paper-2023 what version 1 did), none, mutual-best, '
                         'gale-church, legacy-greedy, or file:PATH with one "i<TAB>j" per line; more through the '
                         'entry-point group sentalign.anchorers')
    g3.add_argument('--anchor-margin', type=float, default=d.anchor_margin,
                    help='an anchor pair must score this much above the second-best match of either sentence (default: %(default)s)')
    g3.add_argument('--anchor-min-words', type=int, default=d.anchor_min_words,
                    help='shortest sentence that can be an anchor (default: %(default)s)')
    g3.add_argument('--max-segment-nodes', type=int, default=d.max_segment_nodes,
                    help='largest segment, in nodes, that is solved: a segment without an anchor inside it, or a whole '
                         'document that is not cut; above it the file fails (default: %(default)s)')
    g2 = parser.add_argument_group('path search')
    g2.add_argument('--pathfinder', default=d.pathfinder,
                    help='path search algorithm (default: %(default)s; more through the entry-point group sentalign.pathfinders)')
    g2.add_argument('--reward', choices=tuple(REWARDS), default=d.reward,
                    help='link reward of the sentalign-dp search. sentalign (default): (ks+kt) * score^(ks+kt-1), a skipped '
                         'sentence earns the cutoff. squared-cosine: score^2, skipping is free (the link weight of '
                         'ChainAlign in an exact search); the default cutoff is then 0.3')
    g2.add_argument('--length-prior', action='store_true',
                    help='with --reward squared-cosine: multiply the reward by the length-ratio prior of ChainAlign '
                         '(for complete translations; hurts when one side has text without a counterpart)')
    g2.add_argument('--pathfinder-opt', dest='pathfinder_options', action=_KeyValue, nargs='+', metavar='NAME=VALUE',
                    help='settings of the chosen path finder, e.g. --pathfinder vecalign --pathfinder-opt '
                         'alignment_max_size=5 del_percentile_frac=0.2')
    g2.add_argument('--path-backend', choices=('auto', 'cython', 'rust', 'python', 'legacy'), default=d.path_backend,
                    help='auto/cython: compiled kernel; rust: the same kernel on several cores (-proc), needs the '
                         'package sentalign-rust; python: readable reference (slow); all give the same results. '
                         'legacy: the 2023 code itself, only with --compat paper-2023 (default: %(default)s)')
    g.add_argument('--batch-texts', type=int, default=None, metavar='N',
                   help='encode many small files together, up to N texts per group (0: one file at a time; '
                        'default 200000 with --compat none, 0 with paper-2023)')
    g.add_argument('--trust-remote-code', action='store_true',
                   help='load an encoder that comes with its own model code; the code is downloaded '
                        'and executed, so use it only for a model you trust')
    g.add_argument('--half', action=argparse.BooleanOptionalAction, default=None,
                   help='encode with the model in half precision (GPU only): about 3 times faster and half the memory, '
                        'vectors differ in the 3rd decimal. Default: on when the device is cuda and --compat none, '
                        'off otherwise; --no-half gives float32 (the numbers of the 2023 paper)')
    g.add_argument('--embedding-cache', default=None, metavar='DIR',
                   help='keep span vectors in DIR and reuse them when the same text is aligned again')


def config_from_args(args: argparse.Namespace) -> AlignConfig:
    names = {f.name for f in dataclasses.fields(AlignConfig)}
    # languages that were not given: the defaults of the dataclass (eng / isl, as version 1)
    languages = ('source_language', 'target_language')
    return AlignConfig(**{k: v for k, v in vars(args).items() if k in names and not (k in languages and v is None)})

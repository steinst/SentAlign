"""What to cite for a run: the papers behind SentAlign and behind each method of other authors that the run uses.

The command line prints the list after every run (``--bibtex`` adds the BibTeX entries); from Python use
``sentalign.citations(**options)`` or ``SentAlign.citations()``.
"""

ENTRIES = {
    'sentalign2': {
        'text': 'Steingrímsson, S. (in preparation). SentAlign 2. [placeholder: the paper has not been published; '
                'this entry will be replaced]',
        'bibtex': '''@misc{sentalign2-placeholder,
    title = {{SentAlign 2}},
    author = {Steingr{\\'i}msson, Stein{\\th}{\\'o}r},
    year = {2026},
    note = {Placeholder: paper in preparation; this entry will be replaced when it is published},
}''',
    },
    'sentalign': {
        'text': 'Steingrímsson, S., Loftsson, H. and Way, A. (2023). SentAlign: Accurate and Scalable Sentence '
                'Alignment. EMNLP 2023: System Demonstrations.',
        'bibtex': '''@inproceedings{sentalign-2023,
    title = {{SentAlign: Accurate and Scalable Sentence Alignment}},
    author = "Steingr{\\'i}msson, Stein{\\th}{\\'o}r and Loftsson, Hrafn and Way, Andy",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    pages = "256--263",
    doi = "10.18653/v1/2023.emnlp-demo.22",
    url = "https://aclanthology.org/2023.emnlp-demo.22/",
}''',
    },
    'labse': {
        'text': 'Feng, F., Yang, Y., Cer, D., Arivazhagan, N. and Wang, W. (2022). Language-agnostic BERT Sentence '
                'Embedding. ACL 2022.',
        'bibtex': '''@inproceedings{feng-etal-2022-language,
    title = "Language-agnostic {BERT} Sentence Embedding",
    author = "Feng, Fangxiaoyu and Yang, Yinfei and Cer, Daniel and Arivazhagan, Naveen and Wang, Wei",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    pages = "878--891",
}''',
    },
    'vecalign': {
        'text': 'Thompson, B. and Koehn, P. (2019). Vecalign: Improved Sentence Alignment in Linear Time and Space. '
                'EMNLP-IJCNLP 2019.',
        'bibtex': '''@inproceedings{thompson-koehn-2019-vecalign,
    title = "{V}ecalign: Improved Sentence Alignment in Linear Time and Space",
    author = "Thompson, Brian and Koehn, Philipp",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    pages = "1342--1348",
}''',
    },
    'chainalign': {
        'text': 'Goldner, O. N. and Gramazio, C. C. (2026). ChainAlign: Cross-Lingual Sentence Alignment via Sparse '
                'Co-linear Chaining. WMT 2026.',
        'bibtex': '''@inproceedings{goldner-gramazio-2026-chainalign,
    title = "{C}hain{A}lign: Cross-Lingual Sentence Alignment via Sparse Co-linear Chaining",
    author = "Goldner, Owen N. and Gramazio, Connor C.",
    booktitle = "Proceedings of the Eleventh Conference on Machine Translation",
    year = "2026",
    publisher = "Association for Computational Linguistics",
    pages = "183--198",
}''',
    },
    'galechurch': {
        'text': 'Gale, W. A. and Church, K. W. (1993). A Program for Aligning Sentences in Bilingual Corpora. '
                'Computational Linguistics 19(1).',
        'bibtex': '''@article{gale-church-1993-program,
    title = "A Program for Aligning Sentences in Bilingual Corpora",
    author = "Gale, William A. and Church, Kenneth W.",
    journal = "Computational Linguistics",
    volume = "19",
    number = "1",
    year = "1993",
    pages = "75--102",
}''',
    },
    'laser': {
        'text': 'Artetxe, M. and Schwenk, H. (2019). Massively Multilingual Sentence Embeddings for Zero-Shot '
                'Cross-Lingual Transfer and Beyond. TACL 7.',
        'bibtex': '''@article{artetxe-schwenk-2019-massively,
    title = "Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond",
    author = "Artetxe, Mikel and Schwenk, Holger",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    pages = "597--610",
}''',
    },
    'laser3': {
        'text': 'Heffernan, K., Çelebi, O. and Schwenk, H. (2022). Bitext Mining Using Distilled Sentence '
                'Representations for Low-Resource Languages. Findings of EMNLP 2022.',
        'bibtex': '''@inproceedings{heffernan-etal-2022-bitext,
    title = "Bitext Mining Using Distilled Sentence Representations for Low-Resource Languages",
    author = "Heffernan, Kevin and {\\c{C}}elebi, Onur and Schwenk, Holger",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
}''',
    },
    'bge-m3': {
        'text': 'Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. and Liu, Z. (2024). BGE M3-Embedding: '
                'Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge '
                'Distillation. arXiv:2402.03216.',
        'bibtex': '''@misc{chen-etal-2024-bge,
    title = {{BGE} {M3}-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
    author = {Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
    year = {2024},
    eprint = {2402.03216},
    archivePrefix = {arXiv},
}''',
    },
    'sbert-multilingual': {
        'text': 'Reimers, N. and Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using '
                'Knowledge Distillation. EMNLP 2020.',
        'bibtex': '''@inproceedings{reimers-gurevych-2020-making,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "4512--4525",
}''',
    },
    'bleualign-data': {
        'text': 'Sennrich, R. and Volk, M. (2010). MT-based Sentence Alignment for OCR-generated Parallel Texts. '
                'AMTA 2010. [the German-French Text+Berg evaluation set]',
        'bibtex': '''@inproceedings{Sennrich2010MTbasedSA,
    title = {{MT-based Sentence Alignment for OCR-generated Parallel Texts}},
    author = "Rico Sennrich and Martin Volk",
    booktitle = "Proceedings of the 9th Conference of the Association for Machine Translation in the Americas: Research Papers",
    address = "Denver, Colorado",
    year = "2010",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2010.amta-papers.14",
}''',
    },
    'parice-data': {
        'text': 'Barkarson, S. and Steingrímsson, S. (2019). Compiling and Filtering ParIce: An English-Icelandic '
                'Parallel Corpus. NoDaLiDa 2019. [the English-Icelandic evaluation set]',
        'bibtex': '''@inproceedings{barkarson-steingrimsson-2019-compiling,
    title = {{Compiling and Filtering {P}ar{I}ce: An {E}nglish-{I}celandic Parallel Corpus}},
    author = "Barkarson, Starka{\\dh}ur and Steingr{\\'i}msson, Stein{\\th}{\\'o}r",
    booktitle = "Proceedings of the 22nd Nordic Conference on Computational Linguistics",
    year = "2019",
    address = "Turku, Finland",
    publisher = {Link{\\"o}ping University Electronic Press},
    pages = "140--145",
}''',
    },
}


def _encoder_keys(cfg):
    """(what it is used for, [entry keys]) of the encoder, or a note when the model is not one we know."""
    name = str(cfg.model_name)
    low = name.lower()
    if cfg.encoder_kind == 'laser':
        return 'the LASER encoder (%s)' % name, ['laser'] + (['laser3'] if low in ('laser2', 'laser3') else [])
    if 'labse' in low:
        return 'the LaBSE encoder', ['labse']
    if 'bge-m3' in low:
        return 'the BGE-M3 encoder', ['bge-m3']
    if 'paraphrase-multilingual' in low:
        return 'the encoder %s' % name, ['sbert-multilingual']
    return 'the encoder %s' % name, None


def citations_for(cfg):
    """[(what it is used for, entry key or None, citation text)] for a configuration, SentAlign first."""
    out = [('SentAlign', 'sentalign2'), ('SentAlign', 'sentalign')]
    scorer_has_encoder = cfg.scorer == 'embedding'
    if scorer_has_encoder:
        what, keys = _encoder_keys(cfg)
        if keys is None:
            out.append((what, None))
        else:
            out.extend((what, k) for k in keys)
    if cfg.pathfinder == 'vecalign':
        out.append(('the Vecalign path search (--pathfinder vecalign)', 'vecalign'))
    elif cfg.pathfinder == 'chain':
        out.append(('the chain path search (--pathfinder chain), a re-implementation after ChainAlign', 'chainalign'))
    elif cfg.pathfinder == 'galechurch':
        out.append(('the Gale-Church path search (--pathfinder galechurch)', 'galechurch'))
    elif cfg.pathfinder != 'sentalign-dp':
        out.append(('the path finder %s' % cfg.pathfinder, None))
    if cfg.pathfinder == 'sentalign-dp' and cfg.reward == 'squared-cosine':
        out.append(('the squared-cosine link reward%s (--reward squared-cosine), from ChainAlign'
                    % (' and the length prior' if cfg.length_prior else ''), 'chainalign'))
    if cfg.anchors == 'gale-church':
        out.append(('the Gale-Church anchors (--anchors gale-church)', 'galechurch'))
    rows, seen = [], set()
    for what, key in out:
        if key is None:
            rows.append((what, None, 'cite the publication of this component (SentAlign does not know it)'))
        elif key not in seen:
            seen.add(key)
            rows.append((what, key, ENTRIES[key]['text']))
    return rows


def format_citations(rows, bibtex=False):
    """The notice printed after a run."""
    lines = ['', 'If you use the results of this run, please cite:']
    for what, key, text in rows:
        lines.append('  [%s] %s' % (what, text))
    if bibtex:
        lines += ['', 'BibTeX:', '']
        lines += [ENTRIES[key]['bibtex'] + '\n' for _, key, _ in rows if key]
    else:
        lines.append('(--bibtex prints the BibTeX entries)')
    return '\n'.join(lines)

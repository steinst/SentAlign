## Evaluation Sets

For convenience we provide the two evaluation sets used to evaluate the alignment tool.

### Bleualign and Text+Berg evaluation set
The bleualign folder contains the manually aligned German-French evaluation set created from the Text+Berg corpus, first used to evaluate Bleualign and commonly used for sentence alignment evaluation since. If you use that evaluation set you should cite the Bleualign paper:

```bibtex
@inproceedings{Sennrich2010MTbasedSA,  
    title = {{MT-based Sentence Alignment for OCR-generated Parallel Texts}},
    author = "Rico Sennrich and Martin Volk",  
    booktitle = "Proceedings of the 9th Conference of the Association for Machine Translation in the Americas: Research Papers",
    address = "Denver, Colorado",
    year = "2010",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2010.amta-papers.14",
}
```

### ParIce evaluation set
The parice folder contains an evaluation set for English-Icelandic sentence alignment from 10 aligned documents in five subcorpora of the ParIce corpus. The evaluation set is [distributed](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/150) under a CC BY 4.0 license. If you use that evaluation set, please consider citing the ParIce paper and the SentAlign paper where the evaluation set was first used.

The ParIce paper:

```bibtex
@inproceedings{barkarson-steingrimsson-2019-compiling,
    title = {{Compiling and Filtering {P}ar{I}ce: An {E}nglish-{I}celandic Parallel Corpus}},
    author = "Barkarson, Starka{\dh}ur and Steingr{\'\i}msson, Stein{\th}{\'o}r",
    booktitle = "Proceedings of the 22nd Nordic Conference on Computational Linguistics",
    month = sep # "{--}" # oct,
    year = "2019",
    address = "Turku, Finland",
    publisher = {Link{\"o}ping University Electronic Press},
    url = "https://aclanthology.org/W19-6115",
    pages = "140--145",
}
```
The SentAlign paper:

```bibtex
@inproceedings{sentalign-2023,
    title = {{SentAlign: Accurate and Scalable Sentence Alignment}},
    author = "Steingrímsson, Steinþór and
      Loftsson, Hrafn  and
      Way, Andy",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
}
```


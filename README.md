# Tamil Deep Learning Awesome List

> A curated catalog of open-source resources for Tamil NLP & AI.

<img src="https://1.bp.blogspot.com/-jImAZD8-kIY/WhwLddVQ0FI/AAAAAAAABmY/cW7pjolPoS4KGb3iXrxikDBgWL3VLAqpwCEwYBhgL/s1600/A%2Btamil%2Btypo%2Bnw.jpg" height="400px" />

The estimated worldwide Tamil population is around 80-85 million which is near to the population of Germany.  
Hence it is crucial to work on NLP for தமிழ். This list will serve as a catalog for all resources related to Tamil NLP.

Note:
- *Please use [GitHub Issues](https://github.com/narVidhai/tamil-nlp-catalog/issues) for queries/feedback or to **contribute** resources/links.*
- *If you find this useful, please [star this on GitHub](https://github.com/narVidhai/tamil-nlp-catalog) to encourage this list to be active.*
- *Share this [awesome website](https://narvidhai.github.io/tamil-nlp-catalog) if you liked it! :-)*

<div id="toc-container">
<hr/>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [**Tools, Libraries, Models**](#tools-libraries-models)
  - [General](#general)
  - [Word Embeddings](#word-embeddings)
  - [Transformers, BERT](#transformers-bert)
  - [Translation](#translation)
  - [Transliteration](#transliteration)
  - [OCR](#ocr)
  - [Speech](#speech)
  - [Miscellaneous](#miscellaneous)
- [**Datasets**](#datasets)
  - [Monolingual Corpus](#monolingual-corpus)
  - [Translation](#translation-1)
  - [Transliteration](#transliteration-1)
  - [Speech, Audio](#speech-audio)
    - [Speech-To-Text](#speech-to-text)
    - [Text-to-Speech (TTS)](#text-to-speech-tts)
    - [Audio](#audio)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Text Classification](#text-classification)
  - [OCR](#ocr-1)
    - [Character-level datasets](#character-level-datasets)
    - [Scene-Text Detection / Recognition](#scene-text-detection--recognition)
  - [Sentiment, Sarcasm, Emotion Analysis](#sentiment-sarcasm-emotion-analysis)
  - [Lexical Resources](#lexical-resources)
  - [Benchmarks](#benchmarks)
  - [Miscellaneous NLP Datasets](#miscellaneous-nlp-datasets)
- [**Other Important Resources**](#other-important-resources)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
</div>

<hr/>

## **Tools, Libraries, Models**

### General

- [iNLTK](https://inltk.readthedocs.io/) - Indian NLP ToolKit
  - Tools for processing and trained models
- [Indic NLP Library](http://anoopkunchukuttan.github.io/indic_nlp_library/)
  - Script-processing tools

Also check Ezhil Foundation's [Awesome-Tamil](https://github.com/Ezhil-Language-Foundation/awesome-tamil) for lot more resources!

### Word Embeddings

- FastText
  - [Wikipedia-based](https://fasttext.cc/docs/en/pretrained-vectors.html) - {2016}
  - [CommonCrawl+Wikipedia](https://fasttext.cc/docs/en/crawl-vectors.html) - {2017}
  - [AI4Bharat IndicFT](https://indicnlp.ai4bharat.org/indicft) - {2020}
  - [Multilingual Aligned](https://github.com/babylonhealth/fastText_multilingual) - {2017}
- [ConceptNet](https://github.com/commonsense/conceptnet-numberbatch)
- [BPEmb: Subword Embeddings](https://nlp.h-its.org/bpemb/) - {2017, [Aligned Multilingual](https://nlp.h-its.org/bpemb/multi/)}
- [PolyGlot](https://sites.google.com/site/rmyeid/projects/polyglot)
- [Facebook MUSE](https://github.com/facebookresearch/MUSE)
- [GeoMM](https://github.com/anoopkunchukuttan/geomm)

### Transformers, BERT

- iNLTK (ULMFit and TransformerXL) - [Tamil](https://github.com/goru001/nlp-for-tamil) | [Tanglish](https://github.com/goru001/nlp-for-tanglish)
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
- [XML RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)
- [AI4Bharat ALBERT](https://indicnlp.ai4bharat.org/indic-bert)
- [Google ELECTRA - TaMillion](https://huggingface.co/monsoon-nlp/tamillion)
- [Google Multilingual T5](https://github.com/google-research/multilingual-t5)
- [Google MuRIL](https://tfhub.dev/google/MuRIL/1)

### Translation

- Moses SMT
  - [Śata-Anuva̅dak](http://www.cfilt.iitb.ac.in/~moses/shata_anuvaadak/)
- NMT
  - [IndicMulti](https://github.com/jerinphilip/ilmulti)
- Transformers
  - [Anuvaad](https://github.com/notAI-tech/Anuvaad)
  - [Facebook Many-to-Many Translation](https://ai.facebook.com/blog/introducing-many-to-many-multilingual-machine-translation)

### Transliteration

- [AI4Bharat Xlit](https://pypi.org/project/ai4bharat-transliteration/)
- [notAI.tech DeepTranslit](https://github.com/notAI-tech/DeepTranslit)
- [Indic Transliteration](https://github.com/sanskrit-coders/indic_transliteration)
- [AksharaMukha](http://aksharamukha.appspot.com/converter) - [API](http://aksharamukha.appspot.com/python)
- LibIndic - [Rule-based](https://github.com/libindic/Transliteration) | [Model-based](https://github.com/libindic/indic-trans)

### OCR

- [Tesseract](https://indic-ocr.github.io/tessdata/)
- [EasyOCR](https://www.jaided.ai/easyocr)

### Speech

- [IIT-M TTS](https://github.com/tshrinivasan/tamil-tts-install)
- [VasuRobo Speech Recognizer](https://github.com/vasurobo/tamil-speech-recognition)

### Miscellaneous

- [Chatbot NER](https://github.com/hellohaptik/chatbot_ner/)
- [Tamilinaiya Spell Checker](https://github.com/tshrinivasan/Tamilinaiya-Spellchecker)

---

## **Datasets**

### Monolingual Corpus

- [WikiDumps](https://dumps.wikimedia.org/tawiki/)
- CommonCrawl
  - [OSCAR Corpus 2019](https://oscar-corpus.com/) - Deduplicated Corpus {226M Tokens, 5.1GB)
  - [WMT Raw 2017](http://data.statmt.org/ngrams/raw/) - CC crawls from 2012-2016
  - [CC-100](http://data.statmt.org/cc-100/) - CC crawls from Jan-Dec 2018
- [WMT News Crawl](http://data.statmt.org/news-crawl/ta/)
- [AI4Bharat IndicCorp](https://indicnlp.ai4bharat.org/corpora/) - {582M}
- [Kaggle Tamil Articles Corpus](https://www.kaggle.com/praveengovi/tamil-language-corpus-for-nlp)
- [LDCIL Standard Text Corpus](https://data.ldcil.org/a-gold-standard-tamil-raw-text-corpus) - Free for students/faculties {11M tokens}
- [EMILLE Corpus](https://www.lancaster.ac.uk/fass/projects/corpus/emille/) - {20M Tokens}
- [Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download/tamil)
- [Project Madurai](https://www.projectmadurai.org/pmworks.html)

### Translation

- [WMT20 NEWS MT Task](http://www.statmt.org/wmt20/translation-task.html) - {2020, Collection of different datasets}
- [CVIT-IIITH](http://preon.iiit.ac.in/~jerin/resources/datasets/) - {[Website](http://preon.iiit.ac.in/~jerin/bhasha/)}
  - Contains data mined from: Press Information Bureau (PIB) and Manathin Kural (MkB)
- [PM India Corpus](https://arxiv.org/pdf/2001.09907.pdf) - {2019, [Download link](http://data.statmt.org/pmindia)}
- [Anuvaad Parallel Corpus](https://github.com/project-anuvaad/anuvaad-parallel-corpus)
- [OPUS Corpus](http://opus.nlpl.eu/) (Search en->ta)
- [Charles University English-Tamil Parallel Corpus](http://ufal.mff.cuni.cz/~ramasamy/parallel/html/)
- [MTurks Crowd-sourced](https://github.com/joshua-decoder/indian-parallel-corpora) - {2012}
- [Facebook WikiMatrix](https://ai.facebook.com/blog/wikimatrix) - {2019, Might be noisy}
- [Facebook CommonCrawl-Matrix](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix) - {2019, Might be noisy)
- [WAT Translation Task](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html) - Other datasets
- [CC Aligned](http://statmt.org/cc-aligned/) - {2020, Collection of Cross-lingual Web-Document Pairs}
- [NLPC-UoM English-Tamil Corpus](https://github.com/nlpc-uom/English-Tamil-Parallel-Corpus) - {2019, 9k sentences}
- [VPT-IL-FIRE2018](http://78.46.86.133/VPT-IL-FIRE2018/)
- [English-Tamil Wiki Titles](http://data.statmt.org/wikititles/v2/wikititles-v2.ta-en.tsv.gz)
- [Corpus by University of Moratuwa](https://github.com/nlpcuom/English-Tamil-Parallel-Corpus)
- [JW300 Corpus](http://opus.nlpl.eu/JW300.php) - Parallel corpus mined from jw.org. Religious text from Jehovah's Witness.
- [IndoWordNet](https://github.com/anoopkunchukuttan/indowordnet_parallel)
- [Indian Language Corpora Initiative](http://sanskrit.jnu.ac.in/ilci/index.jsp) - Available only on request
- TDIL EILMT
  - [Tourism](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1422&lang=en)
  - [Agriculture](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1801&lang=en)
  - [Health](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1789&lang=en)

### Transliteration

- [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina)
- [NEWS2018 Dataset](http://workshop.colips.org/news2018/dataset.html)
- [Thirukkural Transliteration](https://github.com/narVidhai/Thirukkural-transliteration-data)

### Speech, Audio

#### Speech-To-Text

- [Microsoft Speech Corpus](https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e)
- [OpenSLR](http://www.openslr.org/resources.php) - {2020, 9 hours, [Paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.800.pdf)}
- [IARPA Babel](https://catalog.ldc.upenn.edu/LDC2017S13) - {2017, 350 hours}
- [Mozilla CommonVoice](https://commonvoice.mozilla.org/en/datasets) - {2020, 20 hours}
- [Facebook CoVoST](https://github.com/facebookresearch/covost) - {2019, 2 hours}
- [Spoken Tutorial](https://spoken-tutorial.org/) - TODO: Scrape from here

#### Text-to-Speech (TTS)

- [IIT Madras TTS database](https://www.iitm.ac.in/donlab/tts/index.php) - {2020, [Competition](http://tdil-dc.in/ttsapi/ttschallenge2020/)}
- [WikiPron](https://github.com/kylebgorman/wikipron) - Word Pronounciations from Wiki

#### Audio

- [A classification dataset for Tamil music](http://dorienherremans.com/sgmusic) - {2020, [Paper](https://arxiv.org/abs/2009.04459)}

### Named Entity Recognition

- [FIRE2014](http://www.au-kbc.org/nlp/NER-FIRE2014/)
- [FIRE2015 Social Media Text](http://au-kbc.org/nlp/ESM-FIRE2015/) - Tweets
- [WikiAnn](https://elisa-ie.github.io/wikiann) - ([Latest Download Link](https://drive.google.com/drive/folders/1Q-xdT99SeaCghihGa7nRkcXGwRGUIsKN))
- [University of Moratuwa NER](https://github.com/nlpcuom/Sinhala-and-Tamil-NER) - {2019}
- [Tamil Noun Classifier](https://github.com/sarves/Tamil-Noun-Classifier)

### Text Classification

- [IndicGLUE Classification Benchmark](https://indicnlp.ai4bharat.org/indic-glue/)
  - Headline Classification
  - Wikipedia Section Title Classification
  - Wiki Cloze-style Question Answering
- [AI4Bharat News Article Classification](https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset)
- [iNLTK News Articles Classification](https://www.kaggle.com/disisbig/tamil-news-dataset)

### OCR

#### Character-level datasets

- [LipiTK Isolated Handwritten Tamil Character Dataset](http://lipitk.sourceforge.net/datasets/tamilchardata.htm) - {156 characters, 500 samples per char}
- [Tamil Vowels - Scanned Handwritten](https://github.com/anandhkishan/Handwritten-Character-Recognition-using-CNN/tree/master/new_dataset) - {12 vowels, 18 images each}
- [AcchuTamil Printed Characters Dataset](https://github.com/Ezhil-Language-Foundation/acchu-tamilocr-dataset) - {MNIST format}
- [Jaffna University Datasets of printed Tamil characters and documents](http://www.jfn.ac.lk/index.php/data-sets-printed-tamil-characters-printed-documents/)
- [Kalanjiyam: Unconstrained Offline Tamil Handwritten Database](https://kalanjyam.wordpress.com/) - {2016, [Paper](https://link.springer.com/chapter/10.1007/978-3-319-68124-5_24)}

#### Scene-Text Detection / Recognition

- [SynthText](https://github.com/IngleJaya95/SynthTextHindi) - {2019, [Dataset](https://drive.google.com/drive/folders/1fx1D1EW_6_j9tzzXSajM8iQygeMLLMcU)}

### Sentiment, Sarcasm, Emotion Analysis

- [SentiWordNet - SAIL](http://amitavadas.com/SAIL/il_res.html)
- [Dravidian-CodeMix - FIRE2020](https://dravidian-codemix.github.io/2020/datasets.html) - {2020, [Paper](https://www.aclweb.org/anthology/2020.sltu-1.28.pdf)}
- [Twitter Keyword based Emotion Corpus](https://osf.io/48awk/) - {2019}

### Lexical Resources

- [IndoWordNet](http://www.cfilt.iitb.ac.in/indowordnet/)
- [AU-KBC WordNet](http://www.au-kbc.org/nlp/lex_re.html)
- [IIIT-H Word Similarity Database](https://github.com/syedsarfarazakhtar/Word-Similarity-Datasets-for-Indian-Languages)
- [AI4Bharat Word Frequency Lists](https://github.com/AI4Bharat/indicnlp_corpus#text-corpora)
- [MTurks Bilngual Dictionary](https://github.com/AI4Bharat/indicnlp_catalog/issues/21) - {2014}

### Benchmarks

- [Google XTREME](https://github.com/google-research/xtreme)
- [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/)

### Miscellaneous NLP Datasets

- **Natural Language Inference**
  - [XNLI 2019](https://www.gujaratresearchsociety.in/index.php/JGRS/article/view/3426) - Request via email
  - [AI4Bharat Cross-Lingual Sentence Retrieval](https://indicnlp.ai4bharat.org/indic-glue/)
  
- **Dialogue**
  - [Code-Mixed-Dialog 2018](https://github.com/sumanbanerjee1/Code-Mixed-Dialog)

- **Part-Of-Speech (POS) Tagging**
  - [AUKBC-TamilPOSCorpus2016v1](http://www.au-kbc.org/nlp/corpusrelease.html)
  - [ThamizhiPOSt](https://github.com/nlpcuom/ThamizhiPOSt)
  - [Universal Dependencies](https://universaldependencies.org/)
  
- **Information Extraction**  
  (*Can also be event extraction or entity extraction*)
  
  - [EventXtractionIL-FIRE2018](http://78.46.86.133/EventXtractionIL-FIRE2018/)
  - [EDNIL-FIRE2020](https://ednilfire.github.io/ednil/2020/index.html)
  - [CMEE-FIRE2016](http://www.au-kbc.org/nlp/CMEE-FIRE2016/)
  
- **Misc**
  - [Paraphrase Identification - Amrita University-DPIL Corpus](https://nlp.amrita.edu/dpil_cen/index.html)
  - [Anaphora Resolution from Social Media Text - FIRE2020](http://78.46.86.133/SocAnaRes-IL20/)

- **Reasoning**
  - [Cross-lingual Choice of Plausible Alternatives](https://github.com/cambridgeltl/xcopa) (XCOPA)

- MorphAnalysis
  - [AI4Bharat MorphAnalyzer](https://github.com/ai4bharat/indicnlp_corpus#morphanalyzers)
  - [ThamizhiMorph](https://github.com/sarves/thamizhi-morph)

- **Pure Tamil**
  - [Indic to Pure Tamil](https://github.com/narVidhai/Indic-To-Pure-Tamil)
  - [English to Tamil](https://www.kaggle.com/muthua/tamil-loan-words-classification)
  - [Tamil Glossary Dataset](https://osf.io/ngt6v/)
  - [Thamizhi Word Validator](https://github.com/sarves/thamizhi-validator)

- **Image-based**
  - [A Dataset for Troll Classification of TamilMemes](https://github.com/sharduls007/TamilMemes) - {2020, [Paper](https://www.aclweb.org/anthology/2020.wildre-1.2.pdf)}

---

## **Other Important Resources**

- [IndicNLP Catalog](https://github.com/AI4Bharat/indicnlp_catalog) by AI4Bharat
- [The Big Bad NLP Database](https://datasets.quantumstat.com/)

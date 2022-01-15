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
    - [Online translation libraries](#online-translation-libraries)
  - [Transliteration](#transliteration)
  - [OCR](#ocr)
  - [Speech](#speech)
  - [Grammar](#grammar)
  - [Miscellaneous](#miscellaneous)
- [**Datasets**](#datasets)
  - [Monolingual Corpus](#monolingual-corpus)
    - [Government Raw Text](#government-raw-text)
  - [Translation](#translation-1)
    - [Government parallel data](#government-parallel-data)
    - [Papers](#papers)
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
    - [Document OCR](#document-ocr)
  - [Part-Of-Speech (POS) Tagging](#part-of-speech-pos-tagging)
  - [Sentiment, Sarcasm, Emotion and Abuse Analysis](#sentiment-sarcasm-emotion-and-abuse-analysis)
  - [Lexical Resources](#lexical-resources)
  - [Benchmarks](#benchmarks)
  - [Miscellaneous NLP Datasets](#miscellaneous-nlp-datasets)
- [**Other Important Resources**](#other-important-resources)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
</div>

<hr/>

## **Tools, Libraries, Models**

### General

- [iNLTK](https://inltk.readthedocs.io/) (Tools for processing and trained models)
- [Indic NLP Library](http://anoopkunchukuttan.github.io/indic_nlp_library/) (Script-processing tools)

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

- [TranKit](https://github.com/nlp-uoregon/trankit)
- [Multilingual Text2Text](https://github.com/artitw/text2text)
- iNLTK (ULMFit and TransformerXL) - [Tamil](https://github.com/goru001/nlp-for-tamil) | [Tanglish](https://github.com/goru001/nlp-for-tanglish)
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
- [XML RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)
- AI4Bharat: [ALBERT](https://indicnlp.ai4bharat.org/indic-bert), [BART](https://github.com/AI4Bharat/indic-bart)
- [Google ELECTRA - TaMillion](https://huggingface.co/monsoon-nlp/tamillion) - {2020, [Code](https://mapmeld.medium.com/training-bangla-and-tamil-language-bert-models-46d7262b550f)}
- [Google Multilingual T5](https://github.com/google-research/multilingual-t5), [mT6 and DeltaLM](https://github.com/microsoft/unilm/tree/master/deltalm)
- Google MuRIL - {2020, [TF-Hub](https://tfhub.dev/google/MuRIL/1), [HuggingFace](https://huggingface.co/google/muril-base-cased)}

### Translation

- NMT
  - [AI4Bharat IndicTrans](https://indicnlp.ai4bharat.org/indic-trans/) - {2021, [Paper](https://arxiv.org/abs/2104.05596)}
  - [not-AI-Tech Anuvaad](https://github.com/notAI-tech/Anuvaad) - {2020, mT5 model fine-tuned on public datasets}
  - [IIIT-H IndicMulti](https://github.com/jerinphilip/ilmulti)
  - [EasyNMT](https://github.com/UKPLab/EasyNMT) - Collection of open source multilingual NMT models
- Moses SMT
  - [IIT-B Śata-Anuva̅dak](http://www.cfilt.iitb.ac.in/~moses/shata_anuvaadak/)

#### Online translation libraries

- [Python Translators](https://github.com/UlionTse/translators)

### Transliteration

- [AI4Bharat Xlit](https://pypi.org/project/ai4bharat-transliteration/)
- [notAI.tech DeepTranslit](https://github.com/notAI-tech/DeepTranslit)
- [Indic Transliteration](https://github.com/sanskrit-coders/indic_transliteration)
- [AksharaMukha](http://aksharamukha.appspot.com/converter) - [API](http://aksharamukha.appspot.com/python)
- LibIndic - [Rule-based and Model-based](https://github.com/libindic/indic-trans) | [English words](https://github.com/libindic/Transliteration)
- [PolyGlot Transliteration](https://polyglot.readthedocs.io/en/latest/Transliteration.html)
- [EpiTran](https://pypi.org/project/epitran/) - IPA Transliteration
- [Word Phonemizer](https://github.com/bootphon/phonemizer)
- [WikTra](https://twardoch.github.io/wiktra2/) - Tamil Romanizer

### OCR

- [Tesseract](https://indic-ocr.github.io/tessdata/)
- [EasyOCR](https://www.jaided.ai/easyocr)

### Speech

- [Indic Wav2Vec2](https://indicnlp.ai4bharat.org/indicwav2vec/)
- [Vākyānsh ASR](https://github.com/Open-Speech-EkStep/vakyansh-models)
- [IIT-M TTS](https://github.com/tshrinivasan/tamil-tts-install)
- [VasuRobo Speech Recognizer](https://github.com/vasurobo/tamil-speech-recognition)

### Grammar

- [Tamil Prosody (யாப்பிலக்கணம்) Analyzer](https://github.com/virtualvinodh/avalokitam)
- [Google Nisaba (Text Processing Grammar)](https://github.com/google-research/nisaba/blob/main/nisaba/brahmic/README.md)

### Miscellaneous

- [Tamilinaiya Spell Checker](https://github.com/tshrinivasan/Tamilinaiya-Spellchecker)
- [Tamil Language Model and Tokenizer](https://github.com/ravi-annaswamy/tamil_lm_spm_fai) - {2018}
- [Indic POS Tagger](https://github.com/avineshpvs/indic_tagger)
- [Punctuation Restoration](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration) & [Indic-Punct](https://github.com/Open-Speech-EkStep/indic-punct)
- [Number To Words](https://github.com/sutariyaraj/indic-num2words)

---

## **Datasets**

### Monolingual Corpus

- CommonCrawl
  - [OSCAR Corpus 2019](https://oscar-corpus.com/) - Deduplicated Corpus {226M Tokens, 5.1GB)
  - [WMT Raw 2017](http://data.statmt.org/ngrams/raw/) - CC crawls from 2012-2016
  - [CC-100](http://data.statmt.org/cc-100/) - CC crawls from Jan-Dec 2018
- [AI4Bharat IndicCorp](https://indicnlp.ai4bharat.org/corpora/) - {582M}
- [WikiDumps](https://dumps.wikimedia.org/tawiki/)
- [WMT News Crawl](http://data.statmt.org/news-crawl/ta/)
- [Kaggle Tamil Articles Corpus](https://www.kaggle.com/praveengovi/tamil-language-corpus-for-nlp)
- [Dinamalar News Corpus](https://www.kaggle.com/vijayabhaskar96/tamil-news-dataset-19-million-records) - {2009-19, 120k articles}
- [TamilMurasu News Articles](https://www.kaggle.com/vijayabhaskar96/tamil-news-classification-dataset-tamilmurasu) - {2011-19, 127k articles}
- [Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download/tamil)
- [Cholloadai, 2021](https://github.com/vanangamudi/cholloadai-2021) - 72M phrases (not sentences)

#### Government Raw Text

- [LDCIL Standard Text Corpus](https://data.ldcil.org/a-gold-standard-tamil-raw-text-corpus) - Free for students/faculties {11M tokens}
- [EMILLE Corpus](http://www.emille.lancs.ac.uk/) - {20M Tokens, developed [in collaboration with CIIL](http://corpora.ciil.org/)}
- [Project Madurai](https://www.projectmadurai.org/pmworks.html)

### Translation

- [AI4Bharat Samān-Antar](https://indicnlp.ai4bharat.org/samanantar/) {[Paper](https://arxiv.org/abs/2104.05596)}
  - Contains most open source datasets also as of March 2021
- [OPUS Corpus](http://opus.nlpl.eu/) (Search en->ta)
  - Contains [MultiCC Aligned](http://statmt.org/cc-aligned/), [JW300](https://opus.nlpl.eu/JW300-v1.php), [Tanzil](https://opus.nlpl.eu/Tanzil.php), [bible-corpus](https://github.com/christos-c/bible-corpus), [WikiMatrix](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix), and more...
  - Note: CC-Aligned overlaps with [CommonCrawl-Matrix](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix)
- [MultiIndicMT - WAT2021](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html) / [WMT20 NEWS MT Task](http://www.statmt.org/wmt20/translation-task.html#download)
  - Contains [PM India Corpus](http://data.statmt.org/pmindia), [Manathin Kural (CVIT-MkB)](http://preon.iiit.ac.in/~jerin/bhasha/), [NLPC-UoM Corpus](https://github.com/nlpc-uom/English-Tamil-Parallel-Corpus), [Wiki Titles](http://data.statmt.org/wikititles/v2/wikititles-v2.ta-en.tsv.gz), [Charles University EnTam v2.0 Corpus](http://ufal.mff.cuni.cz/~ramasamy/parallel/html/)
- [MTurks Crowd-sourced](https://github.com/joshua-decoder/indian-parallel-corpora) - {2012}
- EkStep Anuvaad
  - [Parallel Corpora](https://github.com/project-anuvaad/anuvaad-parallel-corpus)
  - [Synthetic Corpus](https://github.com/project-anuvaad/parallel-corpus) - Translations generated using Google
- [Tatoeba Wiki Back-translated data](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/Backtranslations.md)
- [IndoWordNet](https://github.com/anoopkunchukuttan/indowordnet_parallel)
- [VPT-IL-FIRE2018](http://78.46.86.133/VPT-IL-FIRE2018/) - 3k verb phrases, available on request

Note: You can also use the [MTData library](https://pypi.org/project/mtdata/) to automatically download parallel data from many of the above sources.

#### Government parallel data

- [Indian Language Corpora Initiative](http://sanskrit.jnu.ac.in/ilci/index.jsp) - Available only on request
- TDIL EILMT
  - [Tourism](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1422&lang=en), [Agriculture](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1801&lang=en), [Health](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1789&lang=en)
  - Mirrored at [NPLT](https://nplt.in/demo/index.php?route=product/category&path=75_59&limit=100)
- Hindi-Tamil ILCI
  - [Parallel Chunked Text Corpus ILCI-II](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=2067&lang=en), [Tourism Text Corpus](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1411), [Agriculture & Entertainment Text Corpus-ILCI II](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1675), [General Text Corpus](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1271), [Health Text Corpus](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1394)
- [Telugu-Tamil General Text Corpus](https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1570)

#### Papers

- [Sinhala-Tamil Parallel Corpus](https://ucsc.cmb.ac.lk/machine-translation-system-sinhala-tamil-language-pair/) - {[Paper1](https://www.aclweb.org/anthology/U14-1018/), [Paper2](https://ieeexplore.ieee.org/document/7980522), Data available on request?, [Test set](https://github.com/nlpc-uom/Sinhala-Tamil-Aligned-Parallel-Corpus)}
- [cEnTam: Creation of a New English-Tamil Corpus, 2020](https://www.aclweb.org/anthology/2020.bucc-1.10.pdf) - Uses OPUS+WMT20 data
- [MIDAS-NMT, 2018](https://github.com/precog-iiitd/MIDAS-NMT-English-Tamil) - Uses OPUS+EnTam data

### Transliteration

- [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina)
- [NEWS2018 Dataset](http://workshop.colips.org/news2018/dataset.html)
- [Microsoft Multi-Indic Mined Corpus](https://github.com/anoopkunchukuttan/indic_transliteration_analysis) - {2021, [Paper](https://www.aclweb.org/anthology/2021.eacl-main.303/)}
- [TRANSLIT: A Large-scale Name Transliteration Resource](https://github.com/fbenites/TRANSLIT) - {2020, [Paper](https://www.aclweb.org/anthology/2020.lrec-1.399.pdf)}
- [ICTA English-Sinhala-Tamil Names](https://www.language.lk/en/resources/code-resources/) - {2009, 10k triplets, SQL format}
- [Thirukkural Transliteration](https://github.com/narVidhai/Thirukkural-transliteration-data) (Old Tamil)

### Speech, Audio

#### Speech-To-Text

- [Ek-Step ULCA ASR dataset](https://github.com/Open-Speech-EkStep/ULCA-asr-dataset-corpus)
- [Microsoft Speech Corpus](https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e)
- [OpenSLR](http://www.openslr.org/resources.php) - {2020, 9 hours, [Paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.800.pdf)}
- [IARPA Babel](https://catalog.ldc.upenn.edu/LDC2017S13) - {2017, 350 hours}
- [Mozilla CommonVoice](https://commonvoice.mozilla.org/en/datasets) - {2020, 20 hours}
- [Facebook CoVoST](https://github.com/facebookresearch/covost) - {2019, 2 hours}
- [Spoken Tutorial](https://spoken-tutorial.org/) - TODO: Scrape from here

#### Text-to-Speech (TTS)

- [IIT Madras TTS database](https://www.iitm.ac.in/donlab/tts/index.php) - {2020, [Competition](http://tdil-dc.in/ttsapi/ttschallenge2020/)}
- [WikiPron](https://github.com/kylebgorman/wikipron) - Word Pronounciations from Wiki
- [LinguaLibre](https://lingualibre.org/datasets/) - Wiktionary-based word corpus
- [SLR65](http://openslr.org/65) - Crowdsourced high-quality Tamil multi-speaker speech dataset

#### Audio

- [A classification dataset for Tamil music](http://dorienherremans.com/sgmusic) - {2020, [Paper](https://arxiv.org/abs/2009.04459)}

### Named Entity Recognition

- [Chatbot NER](https://github.com/hellohaptik/chatbot_ner/)
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
- [TamilMurasu News Articles Classification](https://www.kaggle.com/vijayabhaskar96/tamil-news-classification-dataset-tamilmurasu)
- [Indic Tamil NLP 2018](https://www.kaggle.com/sudalairajkumar/tamil-nlp)
  - Thirukkural Dataset - {Aṟam, Poruḷ, Inbam} classification
  - Movie Review Dataset
  - News Classficaition
- [A Dataset for Troll Classification of TamilMemes, 2020](https://github.com/bharathichezhiyan/TamilMemes)
- [Offensive Language Identification in Dravidian Languages](https://competitions.codalab.org/competitions/27654) - {2020, [Dataset](https://github.com/manikandan-ravikiran/DOSA)}

### OCR

#### Character-level datasets

- [LipiTK Isolated Handwritten Tamil Character Dataset](http://lipitk.sourceforge.net/datasets/tamilchardata.htm) - {156 characters, 500 samples per char}
- [Tamil Vowels - Scanned Handwritten](https://github.com/anandhkishan/Handwritten-Character-Recognition-using-CNN/tree/master/new_dataset) - {12 vowels, 18 images each}
- [AcchuTamil Printed Characters Dataset](https://github.com/Ezhil-Language-Foundation/acchu-tamilocr-dataset) - {MNIST format}
- [Jaffna University Datasets of printed Tamil characters and documents](http://www.jfn.ac.lk/index.php/data-sets-printed-tamil-characters-printed-documents/)
- [Kalanjiyam: Unconstrained Offline Tamil Handwritten Database](https://kalanjyam.wordpress.com/) - {2016, [Paper](https://link.springer.com/chapter/10.1007/978-3-319-68124-5_24)}

#### Scene-Text Detection / Recognition

- [SynthText](https://github.com/IngleJaya95/SynthTextHindi) - {2019, [Dataset](https://drive.google.com/drive/folders/1fx1D1EW_6_j9tzzXSajM8iQygeMLLMcU)}

#### Document OCR

- [Anuvaad OCR Corpus](https://github.com/project-anuvaad/anuvaad-ocr-corpus#tamil)

### Part-Of-Speech (POS) Tagging
- [AUKBC-TamilPOSCorpus2016v1](http://www.au-kbc.org/nlp/corpusrelease.html)
- [ThamizhiPOSt](https://github.com/nlpcuom/ThamizhiPOSt)
- [Universal Dependencies](https://universaldependencies.org/)

### Sentiment, Sarcasm, Emotion and Abuse Analysis

- [SentiWordNet - SAIL](http://amitavadas.com/SAIL/il_res.html)
- [Dravidian-CodeMix - FIRE2020](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset) - {[Competition](https://dravidian-codemix.github.io/2020/datasets.html), [Paper](https://www.aclweb.org/anthology/2020.sltu-1.28.pdf), [TamilMixSentiment](https://github.com/bharathichezhiyan/TamilMixSentiment)}
  - Implementations: [Theedhum Nandrum](https://github.com/oligoglot/theedhum-nandrum)
- [Twitter Keyword based Emotion Corpus](https://osf.io/48awk/) - {2019}
- [ACTSEA: Annotated Corpus for Tamil & Sinhala Emotion Analysis](https://github.com/Jenarthanan14/Tamil-Sinhala-Emotion-Analysis)
- [Tamil 1k Tweets For Binary Sentiment Analysis](https://kracekumar.com/post/tamil_1k_tweets_binary_sentiment/)
- [Hope Speech Dataset, 2020](https://github.com/bharathichezhiyan/HopeEDI) ([Competition](https://competitions.codalab.org/competitions/27653))
- [IIIT-D Abusive Comment Identification, 2021](https://www.kaggle.com/c/iiitd-abuse-detection-challenge/data)

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
  - [AI4Bharat Cross-lingual Semantic Textual Similarity](https://github.com/AI4Bharat/indicnlp_catalog/issues/146) - {2020}
  - [Multilingual Entity-Linking from WikiNews](http://goo.gle/mewsli-dataset) - {2020}
  - [XQA: A Cross-lingual Open-domain Question Answering Dataset](https://github.com/thunlp/XQA) - {2019, [Paper](https://www.aclweb.org/anthology/P19-1227.pdf)}
  
- **Dialogue**
  - [Code-Mixed-Dialog 2018](https://github.com/sumanbanerjee1/Code-Mixed-Dialog)
  
- **Information Extraction**  
  (*Can also be event extraction or entity extraction*)
  
  - [EventXtractionIL-FIRE2018](http://78.46.86.133/EventXtractionIL-FIRE2018/)
  - [EDNIL-FIRE2020](https://ednilfire.github.io/ednil/2020/index.html)
  - [CMEE-FIRE2016](http://www.au-kbc.org/nlp/CMEE-FIRE2016/)
  
- **Misc**
  - [Paraphrase Identification - Amrita University-DPIL Corpus](https://nlp.amrita.edu/dpil_cen/index.html)
  - [Anaphora Resolution from Social Media Text - FIRE2020](http://78.46.86.133/SocAnaRes-IL20/)
  - [MMDravi - Image Captioning and Translation Benchmark, 2019](https://github.com/bharathichezhiyan/multimodalmachinetranslation-Tamil) - Contains manually annotated data for dev & tests from Flickr30k dataset
  - [WIT : Wikipedia-based Image Text Dataset, 2021](https://github.com/google-research-datasets/wit)
  - [AllNewLyrics Dataset - Tamil Song Lyrics](https://github.com/praveenraj0904/tamillyricscorpus) - {2021, [Paper](https://www.aclweb.org/anthology/2021.dravidianlangtech-1.1/)}
  - [TamilPaa Song-Lyrics Dataset, 2020](https://www.kaggle.com/sivaskvs/tamil-songs-lyrics-dataset)

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

---

## **Other Important Resources**

- [IndicNLP Catalog](https://github.com/AI4Bharat/indicnlp_catalog) by AI4Bharat
- [The Big Bad NLP Database](https://datasets.quantumstat.com/)

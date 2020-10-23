# Tamil NLP Awesome List

> A curated catalog of open-source resources for Tamil NLP & AI.

<img src="https://1.bp.blogspot.com/-jImAZD8-kIY/WhwLddVQ0FI/AAAAAAAABmY/cW7pjolPoS4KGb3iXrxikDBgWL3VLAqpwCEwYBhgL/s1600/A%2Btamil%2Btypo%2Bnw.jpg" height="400px" />

The estimated worldwide Tamil population is around 80-85 million which is near to the population of Germany.  
Hence it is crucial to work on NLP for தமிழ். This list will serve as a catalog for all resources related to Tamil NLP.

Note:  
*Please use the "Issues" tab for queries or to contribute resources/links.*

---

## Tools, Libraries & Models

### General

- [iNLTK](https://inltk.readthedocs.io/) - Indian NLP ToolKit
  - Tools for processing and trained models
- [Indic NLP Library](http://anoopkunchukuttan.github.io/indic_nlp_library/)
  - Script-processing tools

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

### Transformers & BERT

- iNLTK (ULMFit and TransformerXL) - [Tamil](https://github.com/goru001/nlp-for-tamil) | [Tanglish](https://github.com/goru001/nlp-for-tanglish)
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
- [XML RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)
- [AI4Bharat ALBERT](https://indicnlp.ai4bharat.org/indic-bert)

### Translation

- Moses SMT
  - [Śata-Anuva̅dak](http://www.cfilt.iitb.ac.in/~moses/shata_anuvaadak/)

### Transliteration

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

---

## Datasets

### Monolingual Corpus

- [WikiDumps](https://dumps.wikimedia.org/tawiki/)
- CommonCrawl
  - [OSCAR Corpus 2019](https://oscar-corpus.com/) - Deduplicated Corpus {226M Tokens, 5.1GB)
  - [WMT Raw 2017](http://data.statmt.org/ngrams/raw/) - CC crawls from 2012-2016
- [WMT News Crawl](http://data.statmt.org/news-crawl/ta/)
- [AI4Bharat IndicCorp](https://indicnlp.ai4bharat.org/corpora/) - {582M}
- [LDCIL Standard Text Corpus](https://data.ldcil.org/a-gold-standard-tamil-raw-text-corpus) - Free for students/faculties {11M tokens}
- [EMILLE Corpus](https://www.lancaster.ac.uk/fass/projects/corpus/emille/) - {20M Tokens}
- [Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download/tamil)

### Translation

- [CVIT-IIITH](http://preon.iiit.ac.in/~jerin/resources/datasets/) - {[Website](http://preon.iiit.ac.in/~jerin/bhasha/)}
  - Contains data mined from: Press Information Bureau (PIB) and Manathin Kural (MkB)
- [PM India Corpus](https://arxiv.org/pdf/2001.09907.pdf) - {2019, [Download link](http://data.statmt.org/pmindia)}
- [OPUS Corpus](http://opus.nlpl.eu/) (Search en->ta)
- [Charles University English-Tamil Parallel Corpus](http://ufal.mff.cuni.cz/~ramasamy/parallel/html/)
- [MTurks Crowd-sourced](https://github.com/joshua-decoder/indian-parallel-corpora) - {2012}
- [Facebook WikiMatrix](https://ai.facebook.com/blog/wikimatrix) - {2019, Might be noisy}
- [Facebook CommonCrawl-Matrix](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix) - {2019, Might be noisy)
- [WAT Translation Task](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html) - Other datasets
- [NLPC-UoM English-Tamil Corpus](https://github.com/nlpc-uom/English-Tamil-Parallel-Corpus) - {2019, 9k sentences}
- [English-Tamil Wiki Titles](http://data.statmt.org/wikititles/v2/wikititles-v2.ta-en.tsv.gz)
- [JW300 Corpus](http://opus.nlpl.eu/JW300.php) - Parallel corpus mined from jw.org. Religious text from Jehovah's Witness.
- [IndoWordNet](https://github.com/anoopkunchukuttan/indowordnet_parallel)
- [Indian Language Corpora Initiative](http://sanskrit.jnu.ac.in/ilci/index.jsp) - Available only on request
- TDIL EILMT
  - [Tourism](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1422&lang=en)
  - [Agriculture](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1801&lang=en)
  - [Health](http://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1789&lang=en)

### Transliteration

- [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina) - {300k pairs}
- [NEWS2018 Dataset](http://workshop.colips.org/news2018/dataset.html)

### Speech

### Speech-To-Text

- [Microsoft Speech Corpus](https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e)
- [OpenSLR](http://www.openslr.org/resources.php) - {2020, 9 hours, [Paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.800.pdf)}
- [Facebook CoVoST](https://github.com/facebookresearch/covost) - {2019, 2 hours}
- [Spoken Tutorial](https://spoken-tutorial.org/) - TODO: Scrape from here

### Text-to-Speech (TTS)

- [IIT Madras TTS database](https://www.iitm.ac.in/donlab/tts/index.php) - {2020, [Competition](http://tdil-dc.in/ttsapi/ttschallenge2020/)}
- [WikiPron](https://github.com/kylebgorman/wikipron) - Word Pronounciations from Wiki

### Named Entity Recognition

- [FIRE2014](http://www.au-kbc.org/nlp/NER-FIRE2014/)
- [WikiAnn](https://elisa-ie.github.io/wikiann) - ([Latest Download Link](https://drive.google.com/drive/folders/1Q-xdT99SeaCghihGa7nRkcXGwRGUIsKN))

### Text Classification

- [IndicGLUE Classification Benchmark](https://indicnlp.ai4bharat.org/indic-glue/)
  - Headline Classification
  - Wikipedia Section Title Classification
  - Wiki Cloze-style Question Answering
- [AI4Bharat News Article Classification](https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset)

### Sentiment, Sarcasm, Emotion Analysis

- [SentiWordNet - SAIL](http://amitavadas.com/SAIL/il_res.html)

### Lexical Resources

- [IndoWordNet](http://www.cfilt.iitb.ac.in/indowordnet/)
- [IIIT-H Word Similarity Database](https://github.com/syedsarfarazakhtar/Word-Similarity-Datasets-for-Indian-Languages)
- [AI4Bharat Word Frequency Lists](https://github.com/AI4Bharat/indicnlp_corpus#text-corpora)

### Miscellaneous NLP Datasets

- **Part-Of-Speech (POS) Tagging**
  - [Universal Dependencies](https://universaldependencies.org/)
  
- **Information Extraction**
  - [EventXtractionIL-FIRE2018](http://78.46.86.133/EventXtractionIL-FIRE2018/)
  - [EDNIL-FIRE2020](https://ednilfire.github.io/ednil/2020/index.html)
  
- **Paraphrase Identification**
  - [Amrita University-DPIL Corpus](https://nlp.amrita.edu/dpil_cen/index.html)
  
- MorphAnalyzers - {[AI4Bharat](https://github.com/ai4bharat/indicnlp_corpus#morphanalyzers)}

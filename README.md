# CSCE 578 Final Project

*Note: A reference for dealing with JSON transcripts can be found in [this code](clustering/Glaeser_assignment02.py).*

## Quick Links

* [Project Proposal](https://www.overleaf.com/read/nmbpfkzvzjgj); also [in this repo](Glaeser,Quan_Proposal.pdf)
* [Project Write-up](https://www.overleaf.com/read/nmbpfkzvzjgj)

## Setup and Dependencies

* Python 3.7 (except for the scripts in the utilities folder, which use 2.7).  
* The Sentence Complexity part uses the Stanford Parser, which you can download as follows:
```
cd ~/Downloads
wget http://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip
unzip stanford-parser-full-2018-10-17.zip
wget https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
```

Finally, set the path by typing the following into your command prompt:
```
export PROJECT_PATH="[path to this repo]"
```

## Utilities

*Python 2.7*

* README
* `groupme-fetch.py`: script for getting GroupMe transcripts
* `newest-id.py`: used by `groupme-fetch.py`
* `simple-transcript.py`: prints a human-readable version of a given transcript; same functionality can be found in `clustering/Glaeser_assignment02.py`
* `stopwords.txt`: standard list of stopwords (provided by Dr. Duncan Buell)

## Clustering

*Python 3.7*

* [Write-up explaining original code](https://www.overleaf.com/read/cwzdnysgycvf)
* Python code

## Sentence Complexity

*Python 3.7*

* [Write-up explaining original code](https://www.overleaf.com/read/zczwcrsfwjqk)
* `Glaeser_assignment03.ipynb`: full code

There are also files containing submodules of the full code:  
* `read_sentences.ipynb`
* `parse_sentences.ipynb`
* `results.ipynb`

## Sentiment Analysis

*Python 3.7*

## Topic Modelling

*Python 3.7*

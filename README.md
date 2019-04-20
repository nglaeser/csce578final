# CSCE 578 Final Project

*Note: A reference for dealing with JSON transcripts can be found in [this code](clustering/Glaeser_assignment02.py)*

## Quick Links

* [Project Proposal](https://www.overleaf.com/read/nmbpfkzvzjgj); also [in this repo](Glaeser,Quan_Proposal.pdf)
* [Project Write-up](https://www.overleaf.com/read/nmbpfkzvzjgj)

## Setup and Dependencies

This project utilizes Python 3.  

In several files, I also use environment variables to get the path to transcript files, etc. Set up your path by typing the following into your command prompt:
```
export PROJECT_PATH="[your project's directory]"
```

## Utilities

* README
* `groupme-fetch.py`: script for getting GroupMe transcripts
* `newest-id.py`: used by `groupme-fetch.py`
* `simple-transcript.py`: prints a human-readable version of a given transcript; same functionality can be found in `clustering/Glaeser\_assignment02.py`
* `stopwords.txt`: standard list of stopwords (provided by Duncan Buell)

## Clustering

* [Write-up explaining original code](https://www.overleaf.com/read/cwzdnysgycvf)
* Python and iPython versions of code

## Sentence Complexity

* [Write-up explaining original code](https://www.overleaf.com/read/zczwcrsfwjqk)
* `Glaeser_assignment03.ipynb`: full code

There are also files containing submodules of the full code:  
* `read_sentences.ipynb`
* `parse_sentences.ipynb`
* `results.ipynb`

## Sentiment Analysis

## Topic Modelling

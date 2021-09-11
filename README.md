# CSCE 578 Final Project

*Note 2: This repo has been archived.*

*Note: A reference for dealing with JSON transcripts can be found in [this code](clustering/Glaeser_assignment02.py).*

This project runs several types of text analysis on the corpus of messages from a GroupMe chat.

## Quick Links

* [Project Proposal](https://www.overleaf.com/read/nmbpfkzvzjgj); also [in this repo](Glaeser,Quan_Proposal.pdf)
* [Project Write-up](https://www.overleaf.com/read/nmbpfkzvzjgj)

## Overall Setup and Dependencies

* Python 3.7 (except for the scripts in the utilities folder, which use 2.7).  
* Make sure your transcript is up-to-date (see [utilities/README.md](utilities/README.md))

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

Cluster users by the content of their messages.

* [Write-up explaining original code](https://www.overleaf.com/read/cwzdnysgycvf)
* Python code

## Sentence Complexity

*Python 3.7*

Determine the sentence complexity of each user's messages.

* [Write-up explaining original code](https://www.overleaf.com/read/zczwcrsfwjqk)
* `Glaeser_assignment03.ipynb` and `Glaeser_assignment03.py`: full code

There are also files containing submodules of the full code:  
* `read_sentences.ipynb`
* `parse_sentences.ipynb`
* `results.ipynb`

## Sentiment Analysis

*Python 3.7*

Analyze the mood of the messages in the chat over time.

## Topic Modelling

*Python 3.7*

Obtain topics that describe the content of the chat in a certain time frame.

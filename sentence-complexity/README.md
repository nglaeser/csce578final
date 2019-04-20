## Usage

```
cd ~/Downloads/stanford-corenlp-full-2018-10-05
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 & 
cd $PROJECT_PATH
python Glaeser_assignment03.py
```

## Improvements to be made

* Double-check that the depth works correctly
* Check out *The Washington Post*'s sentence complexity measures

1) python groupme-fetch.py [output name] [access token] --> OUTPUT FILE temp-transcript-foo.json
***note: you may have to upgrade the cryptography package***
sudo -H pip install cryptography --upgrade --ignore-installed

python groupme-fetch.py GROUPID ACCESSTOKEN newest $(python newest-id.py transcript-GROUPID.json)

# GoupMe Fetch Scripts

By @cdzombak, see the repo [here](https://github.com/cdzombak/groupme-tools). It explains how to find your group's access token and group ID, which are necessary arguments to the script.

## Usage

Get a new transcript:  
```
python groupme-fetch.py GROUPID ACCESSTOKEN
```

Update a current transcript:  
```
python groupme-fetch.py GROUPID ACCESSTOKEN newest $(python newest-id.py transcript-GROUPID.json)
```

You can get a help message using the `-h` or `--help` flags.

## Dependencies

* Requests: `pip install requests`  
* Python 2.7

You may need to upgrade your cryptography package:
```
sudo -H pip install cryptography --upgrade --ignore-installed
```

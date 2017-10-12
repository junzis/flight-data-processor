# Flight Data Processor

This is a python library to process and analyze flight data (e.g. from decoded ADS-B messages). Following functions and algorithms are implemented:

- Extract continuous full or partial flight path data
  - Unsupervised Machine Learning, Clustering using DBSCAN
- Smoothing, filtering, and interpolating flight data
  - Spline filtering
  - Weighted average filtering
  - Time-based weighted average filtering
- Segmenting flight into different phases:
  - using Fuzzy Logic with data interpolation methods
  - supporting phases: ground, climb, descend, cruise, and level flight

# Required python library
- scipy
- scikit-learn
- skfuzzy
- pymongo (version 2.x) (for extracting data from database)

# Examples

## 1. Flight clustering

1. install MongoDB

2. import the sample scattered flight data

    ```bash
    $ mongoimport -d test_db -c positions --type csv \
                  --file data/sample_adsb_decoded.csv --headerline
    ```

3. extract flight from ADS-B positions

    ```bash
    $ python flightextract.py --db test_db --inColl positions --outColl flights
    ```

## 2. Fuzzy segmentation
You can use previously created collection in MongoDB. Or, using provided pickled data, run:

```bash
$ python test_phases.py
```

The essential code to indentify the flight phases is:
```python
import flightphase
flightphase.fuzzylabels(times, alts, spds, rocs)
```

## 3. View flights

Use the same previously created MongoDB collection:

```bash
$ python flightview.py --db test_db --coll flights
```


## Screen shots
### example flight phase identification
![flight phases](data/images/phase.png?raw=true)

### example fuzzy logic membership functions
![fuzzy logic membership](data/images/membership.png?raw=true)

### example flight viewer
![flight viewer](data/images/flightview.png?raw=true)

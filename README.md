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

# Paper and citation

The source code of this repository complements the following publication:

https://arc.aiaa.org/doi/10.2514/1.I010520

If you use the code for your research, please cite:
```
@article{sun2017flight,
  title={Flight Extraction and Phase Identification for Large Automatic Dependent Surveillance--Broadcast Datasets},
  author={Sun, Junzi and Ellerbroek, Joost and Hoekstra, Jacco},
  journal={Journal of Aerospace Information Systems},
  pages={1--6},
  year={2017},
  publisher={American Institute of Aeronautics and Astronautics}
}
```

# Required software
- Python 3.x
- MongoDB 3
- Dependent Python libraries
  - scipy
  - scikit-learn
  - skfuzzy
  - pymongo

# Code examples

## 1. Flight clustering

1. install MongoDB

2. extract flight from ADS-B positions

    ```bash
    $ python flightextract.py --csv data/sample_adsb_decoded.csv --db test_db --coll flights
    ```

## 2. Fuzzy segmentation
You can use previously created collection in MongoDB. Or, using provided pickled data, run:

```bash
$ python test_phases.py
```

The essential code to identify the flight phases is:

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

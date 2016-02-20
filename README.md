# Flight Data Processor

This is a python library to process and analyze flight data (e.g. from decoded ADS-B messages). Following functions and algorithms are implemneted:

- Extract continuous full or partial flight path data
    - Unsupervised Machine Learning, Clustering (DBSCAN, BIRCH, etc)
- Smoothing, filtering, and interpolating flight data
    - Kalman filter
    - Savitzky Golay filter
    - Splines
- Segmentinh flight data into different phases of ground, climb, descend, and cruise 
    - using Fuzzy Logic with data interpolation methods

## required python library
- scipy
- scikit-learn
- skfuzzy
- pymongo (version 2.x) (for extracting data from database) 

## Some sample results of those scripts
### Fuzzy logic member functions
![fuzzy_mfs](https://cloud.githubusercontent.com/assets/9550577/9793429/a989b30e-57e5-11e5-8211-9f8abaa00123.png)

### Segmention of flight data (without data interpolation)
![segments-original-data](https://cloud.githubusercontent.com/assets/9550577/9793433/b0e7ae44-57e5-11e5-9eea-cd2e26dee4a3.png)

### Segmention of flight data (with data interpolation)
![segments-with-data-interpolation](https://cloud.githubusercontent.com/assets/9550577/9793436/b36833f0-57e5-11e5-9c8f-182cf99a249f.png)

### Flight Viewer script
![flight-view-1](https://cloud.githubusercontent.com/assets/9550577/9844539/59295a86-5ac4-11e5-9b96-9ab881ce376b.png)

![flight-view-2](https://cloud.githubusercontent.com/assets/9550577/9844541/5cd36866-5ac4-11e5-8c1e-5cc5fa9d1c0e.png)

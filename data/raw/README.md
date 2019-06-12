## data/raw folder

Contains the raw (unmodified) data files we used in our analysis.

Content:

* **block-groups.csv**: original Eviction Lab data we used in our analysis. It contains data at the block group level for the state of Illinois. It was directly downloaded from the IL folder in https://data-downloads.evictionlab.org/
* **block-groups_2017_acs-only.csv**: ACS data at the block group level, 2017 only, containing the ACS features we included. We used it for our final prediction outcome. It was downloaded using the ACS API from a Jupyter Notebook, but due to documentation and versioning mistakes we could not save the notebook that downloaded this data
* **block-groups.geojson**: spatial data with polygons for all the block groups in Illinois. It also contains the Eviction Lab data as features. We used for plotting the maps we presented in the final report and in the project presentations. It was directly downloaded from the IL folder in https://data-downloads.evictionlab.org/

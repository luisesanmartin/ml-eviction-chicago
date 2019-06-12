## notebooks

This folder contains the Jupyter Notebooks we used.

Content:

* **archives**: contains notebooks used for specific tasks, but not part of the ML pipeline
* **Bias_and_Fairness_Analysis.ipynb**: bias and fairness analysis of our final results
* **Data exploration - block groups.ipynb**: initial data exploration of the Eviction Lab dataset for the Cook County block groups only
* **Feature Generation.ipynb**: generation of all the final features we used in our supervised learning analysis. It produces the file "block-groups_2012-2016_with-acs_with-gen-features.csv" from the data/work folder
* **Final_Results_Graph.ipynb**: generation of the final list and map of block groups to intervene on. It produces the files "final_blocks_list.csv" and "final_map.png" from the output folder
* **get_acs_data**: downloads and stores the ACS data files we use. It produces the file "block-groups_2012-2016_with-acs.csv" from the data/work folder
* **ML-analysis.ipynb**: final supervised learning analysis of our project. It produces the files "evaluation_table.csv", "selected_models.png", "final_predictions.csv" and "final_model.txt" from the outputs folder
* **ML-analysis.py**: Python script generated from a direct conversion of the "ML-analysis.ipynb" notebook. It was used to run the analysis in the RCC server
* **ML_analysis_sbatch.err**: warnings and errors outputs generated from running "ML-analysis.py" in the RCC server
* **ML_analysis_sbatch.out**: on-screen printed messages generated from running "ML-analysis.py" in the RCC server
* **run.sbatch**: sbatch file to run the "ML-analysis.py" file in the RCC server

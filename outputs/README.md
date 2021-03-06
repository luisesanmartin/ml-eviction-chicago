## outputs

This folder contains the final results files we generated in our analysis.

Contents:

* **archives**: contains files of previous results or files generated for a certain purpose (especially for data visualizations included in our final report), but not part of the analysis pipeline
* **evaluation_table.csv**: performance metrics for all the models we generated applied to all the datasets we used. It is generated by the "ML-analysis" notebook
* **final_blocks_list.csv**: total list of block groups, containing a dummy with the ones our analysis recommends intervening on. It is generated by the "Final_Results_Graph.ipynb" notebook
* **final_map.png**: map with the final list of block groups our analysis recommends intervening on. It is generated by the "Final_Results_Graph.ipynb" notebook
* **final_model.txt**: text file with the final model we used to generate our final results. It is generated by the "Final_Results_Graph.ipynb" notebook
* **final_predictions.csv**: list of block groups and their final predicted scores. It is generated by the "ML-analysis" notebook
* **selected_models.png**: graph of the performance of the models with the best average preferred metric (precision at 10%) for each type of classifier used. It is generated by the "ML-analysis" notebook

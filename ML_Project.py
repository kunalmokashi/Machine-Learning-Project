import pandas
import numpy as np
import Chow_Liu


if __name__ == '__main__':
	df = pandas.read_csv("ML1/ML1/Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv", sep='\t', encoding = "utf-8")
	df.to_csv("ML1/ML1/CleanedDataset.csv")
	
	# preprocess data
	dataframe_clean = pandas.read_csv("ML1/ML1/CleanedDataset2.csv", encoding="utf-8")

    # get dependency tree by running the chow liu algorithm.
	MST = Chow_Liu.run_chow_liu(dataframe_clean)
	print("MST - ", MST)
	
	# do the inference based on the MST obtained above.
	predict_missing_data(MST, dataframe_clean)



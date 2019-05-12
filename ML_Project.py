import pandas
import numpy as np
import Chow_Liu
import Preprocessing_rev1
import ast
from collections import defaultdict

def infer_class_label(MST_graph, tuples, index, initial_index, processed_data, prob_dict, feature_num_of_classes_dict, feature_name_with_index):

    #for the given index to be predicted, get the adjacency list from the MST.
    neighbors = MST_graph.__getneighbors__(index)
    MST_graph.visited[index] = True
    for neighbor in neighbors:
        if MST_graph.visited[neighbor] == False:
            infer_class_label(MST_graph, tuples, neighbor, initial_index, processed_data, prob_dict, feature_num_of_classes_dict, feature_name_with_index)

    if index == initial_index:
        num_classes = feature_num_of_classes_dict[index]
        column_1 = list(processed_data.iloc[:, index])
        max_class_label = 1
        max_prob = 0
        for class_label in range(num_classes):
            p = 1
            for neighbor in neighbors:
                neighbor_value = tuples[neighbor + 1]
                column_neighbor = list(processed_data.iloc[:, neighbor])
                col_idx_nei=tuple(zip(column_1, column_neighbor))
                temp=(col_idx_nei.count((class_label + 1,neighbor_value)))/len(col_idx_nei)
                p = p * (temp / prob_dict[neighbor])
#            prob = p * np.prod(np.array(list(prob_dict.values()),dtype=np.float32))
            if p > max_prob:
                max_prob = p
                max_class_label = class_label + 1
        return max_class_label, max_prob
    else:
        value = tuples[index + 1]
        if not np.isnan(value):
            column_1 = list(processed_data.iloc[:, index])
            if len(neighbors) == 0:
                p = column_1.count(value) / len(column_1)
            else:
                p = 1
                for neighbor in neighbors:
                    neighbor_value = tuples[neighbor + 1]
                    column_neighbor = list(processed_data.iloc[:, neighbor])
                    col_idx_nei=tuple(zip(column_1,column_neighbor))
                    temp=(col_idx_nei.count((value,neighbor_value)))/len(col_idx_nei)
                    if temp == 0:
						# joint distribution is zero, take independent probability.
                        p = p * (column_1.count(value) / len(column_1))
                    else:
                        p = p * (temp/ prob_dict[neighbor])
        else:
            num_classes = feature_num_of_classes_dict[index]
            column_1 = list(processed_data.iloc[:,index])
            probs = []
            for class_label in range(num_classes):
                if len(neighbors) == 0:
                    probs.append(column_1.count(class_label + 1) / len(column_1))
                else:
                    p_inside = 1
                    for neighbor in neighbors:
                        neighbor_value = tuples[neighbor + 1]
                        column_neighbor = list(processed_data.iloc[:, neighbor])
                        col_idx_nei=tuple(zip(column_1,column_neighbor))
                        temp=(col_idx_nei.count((class_label + 1,neighbor_value)))/len(col_idx_nei)
                        if temp == 0:
	                        p_inside = p_inside * (column_1.count(class_label + 1) / len(column_1))
                        else:
                            p_inside = p_inside * (temp / prob_dict[neighbor])
                    probs.append(p_inside)
            p = max(probs)
        prob_dict[index] = p

def conv_str_dict(dat):
    _dict={}
    temp=dat.strip('}{').split(', ')
    temp2=[]
    temp2.extend(te.split(': ') for te in temp)
    for t in temp2:
        _dict[t[0]]=int(t[1])
    return _dict

def predict_missing_data(MST, processed_data, codebook):
    print("predict missing data")
    feature_names = list(codebook.iloc[:, 0])
    print(feature_names)

    #find out how many classes each feature has.
    feature_num_of_classes_dict = {}
    feature_name_with_index = {}
    for index, feature in enumerate(feature_names):
        feature_num_of_classes_dict[index] = len(ast.literal_eval(codebook.iloc[:,1][index]))
        feature_name_with_index[index] = feature
    updated_df = processed_data.copy()
    i= 0
    for tuples in processed_data.itertuples():
        indices = np.where(np.isnan(tuples))
        indices = [a - 1 for a in indices]
        if not len(indices[0]) == 0:
            for index in indices[0]:
                prob_dict = defaultdict(float)
                max_class_label, max_prob = infer_class_label(MST, tuples, index, index, processed_data, prob_dict, feature_num_of_classes_dict, feature_name_with_index)
                updated_df.iloc[i, index] = max_class_label
        i+=1
    updated_df.to_csv("ML1/ML1/PredictedData2.csv")

if __name__ == '__main__':
    df = pandas.read_csv("ML1/ML1/Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv", sep='\t', encoding = "utf-8")
    df.to_csv("ML1/ML1/CleanedDataset.csv")

    # preprocess data
    processed_data, codebook = Preprocessing_rev1.preprocess_data("ML1/ML1/CleanedDataset.csv")
    print("Data pre processing completed.")
    processed_data = pandas.read_csv("ML1/ML1/ProcessedData.csv", encoding="utf-8")
    processed_data = processed_data.drop(processed_data.columns[[0]], axis = 1)
    codebook = pandas.read_csv("ML1/ML1/CodebookData.csv", encoding="utf-8")

    # get dependency tree by running the chow liu algorithm.
    print("Building model")
    no_of_vertices = len(processed_data.columns)
    MST = Chow_Liu.run_chow_liu(processed_data)
    print(MST)
    MST_graph = Chow_Liu.Graph(no_of_vertices)
    for key, value in MST.items():
        MST_graph.add_edge(key[0], key[1], value, True)

    # do the inference based on the MST obtained above.
    print("Number of vertices in the tree - ", len(MST_graph.graph))
    predict_missing_data(MST_graph, processed_data, codebook)



# cleaning the log results
from eventclf.attendance_classification import get_dic_models, get_models_evaluation
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from eventclf.preprocessing import rich_analyzer_textual
from eventclf.w2v_model import MaxEmbeddingVectorizer, SumEmbeddingVectorizer, \
    MixEmbeddingVectorizer
import gensim.downloader
import pandas as pd

#supress sklearn warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

events = ['Creamfields', 'vfestival']
events_date = [[datetime(2016, 8, 25), datetime(2016, 8, 26),
                datetime(2016, 8, 27), datetime(2016, 8, 28)],
               [datetime(2016, 8, 20), datetime(2016, 8, 21)]]

tasks = [1, 2, 3]
groups_feat = ['all']
w2v_vectors = gensim.downloader.load('word2vec-google-news-300')

# dics for the results
dic_models = get_dic_models(allalg=True)
vect_textual = CountVectorizer(tokenizer=rich_analyzer_textual)

vects = {"max": MaxEmbeddingVectorizer(w2v_vectors),
         "sum": SumEmbeddingVectorizer(w2v_vectors),
         "mix": MixEmbeddingVectorizer(w2v_vectors)}

log_result_rq1 = []
print('RQ1 started!')
for idx, event in enumerate(events):
    for task in tasks:
        for group in groups_feat:
            for model_name in dic_models.keys():
                for vect_name in vects.keys():
                    result = get_models_evaluation(dic_models[model_name], idx,
                                                   task, group, vect_textual,
                                                   vects[vect_name])
                    ln = "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s" % (
                    event, task, group, model_name, ("%s_count" % vect_name),
                    result[0], result[2], result[3], result[4], result[5])
                    log_result_rq1.append(ln)
print('RQ1 finished!')

# mount datagrid
dic_result_rq1 = {}
dic_result_rq1['Dataset'] = [x.split(';')[0] for x in log_result_rq1]
dic_result_rq1['Task'] = [x.split(';')[1] for x in log_result_rq1]
dic_result_rq1['Group'] = [x.split(';')[2] for x in log_result_rq1]
dic_result_rq1['Model'] = [x.split(';')[3] for x in log_result_rq1]
dic_result_rq1['Vectorizer'] = [x.split(';')[4] for x in log_result_rq1]
dic_result_rq1['Accuracy'] = [x.split(';')[5] for x in log_result_rq1]
dic_result_rq1['Precision'] = [x.split(';')[6] for x in log_result_rq1]
dic_result_rq1['Recall'] = [x.split(';')[7] for x in log_result_rq1]
dic_result_rq1['F1_score'] = [x.split(';')[8] for x in log_result_rq1]
dic_result_rq1['Dimentions'] = [x.split(';')[9] for x in log_result_rq1]
_df_general_accuracy = pd.DataFrame(dic_result_rq1)

print(_df_general_accuracy)

with open('../results/r1_both_gn_gbt2.csv', 'w') as f:
    _df_general_accuracy.to_csv(f, sep='\t')
print('file saved')

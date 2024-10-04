# cleaning the log results
import pandas as pd
from eventclf.attendance_classification import get_dic_models, get_models_evaluation
from eventclf.preprocessing import rich_analyzer_textual
from sklearn.feature_extraction.text import CountVectorizer
from eventclf.w2v_model import SumEmbeddingVectorizer
import gensim.downloader


def run_feat_analysis(path="../results/"):
    events = ['Creamfields', 'vfestival']
    tasks = [1,2,3]
    dic_models = get_dic_models(allalg=True)
    w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
    vect_w2v = SumEmbeddingVectorizer(w2v_vectors)
    vect_textual = CountVectorizer(tokenizer=rich_analyzer_textual)

    for individual_evaluation in [True, False]:
        if individual_evaluation:
            groups_feat = ['all', 'text', 'media', 'social', 'temporal']
        else:
            groups_feat = [ 'all', '-text', '-media', '-social', '-temporal']

        log_result_rq1 = []
        print('RQ2 started!')
        for idx, event in enumerate(events):
            for task in tasks:
                for group in groups_feat:
                    for model_name in dic_models.keys():
                        #for vect_name in vects.keys():
                        result = get_models_evaluation(dic_models[model_name], idx, task, group, vect_textual, vect_w2v)
                        ln = "%s;%s;%s;%s;%s;%s" % (event, task, group, model_name, 'sum|count', result[0])
                        log_result_rq1.append(ln)
                        print(ln)

        # mount datagrid
        dic_result_rq1 = {}
        dic_result_rq1['Dataset'] = [x.split(';')[0] for x in log_result_rq1]
        dic_result_rq1['Task'] = [x.split(';')[1] for x in log_result_rq1]
        dic_result_rq1['Group'] = [x.split(';')[2] for x in log_result_rq1]
        dic_result_rq1['Model'] = [x.split(';')[3] for x in log_result_rq1]
        dic_result_rq1['Vectorizer'] = [x.split(';')[4] for x in log_result_rq1]
        dic_result_rq1['Accuracy'] = [x.split(';')[5] for x in log_result_rq1]
        df_group_accuracy = pd.DataFrame(dic_result_rq1)

        if individual_evaluation:
            print("save individual evaluation")
            with open(path+'r2_ind_both_gn.csv', 'wb') as f:
                df_group_accuracy.to_csv(f, sep='\t')

        else:
            print("save excluded group evaluation")
            with open(path+'r2_exc_both_gn.csv', 'wb') as f:
                df_group_accuracy.to_csv(f, sep='\t')


def main():
    run_feat_analysis()


if __name__ == "__main__":
    main()
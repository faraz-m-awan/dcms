# Event attendance prediction classifier

Code for answering the following RQ:
- RQ1. How accurate is our event attendance prediction classifier?
  ```
  from eventclf.evaluate_models import eval_model
  eval_model(path_to_results_file)
  ```
- RQ2. What features groups help most to attain high prediction accuracy?
  ```
  from eventclf.evaluate_features import run_feat_analysis
  run_feat_analysis(path_to_results_file)
  ```
  
Find the results in the results folder.
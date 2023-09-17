# datasetclassification

## Structure
### src/corpus_generation
Here you will find scripts for generating a labeled corpus using Wikipedia and Dewey Decimal Classification System 
as a mapping scheme. The main script to use it ```corpus_gen_category_mapping.py```
If all mapping files are present ```corpus_explore_dash.py``` launches a dash application which visualizes the mapping distribution
![image](https://github.com/adrian1703/datasetclassification/assets/65605180/aef205f5-e3e0-4ab0-a5dd-d9eefc85228d)

### src/model
This section contains the code for the Supervised Topic Model, SVM extension. Additionally here are also helper functions to 
employ these models aswell as FastText using k-fold. 

### src/eval_scripts
Here you will find script used for evaluation of the models and the constructed corpus aswell as the script used to 
evaluate their performance on the 20NewsGroupsDataset

## Application Models
1. Construct a corpus
2. Construct the models
   ```python
    import sys
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    
    import model.stm_model_k_fold as stm
    import model.svm_model_k_fold as svm
    import model.fasttext_k_fold as ft
    
    ddc_classes_to_include = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,570,580,590,610,630,650,670,680,710,720,730,740,780,790]
    #%%
    
    
    print("gen stm model")
    src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
    res_stm, stm_models = stm.k_fold(src_dir, {"ranking_c" : lambda a,b: a * b / 100}, ddc_classes=ddc_classes_to_include, early_break=True)
    stm_model = stm_models[0]
    print("gen svm extension")
    res_svm, models_svm = svm.k_fold_svm(stm_models, src_dir, ddc_classes_to_include,early_break=False, svm_C=0.1,  train_chunk=30000)
    print("gen fasttext")
    src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
    res_fasttext, models_fasttext = ft.fasttext_k_fold(src_dir,ddc_classes_to_include, epochs=5, ngrams=2, dim=100, lr=0.6,early_break=True)
    
    stm_model = stm_models[0]
    svm_model = models_svm[0][0]
    svm_scaler = models_svm[0][1]
    ft_model = models_fasttext[0]
3. Classify data
   ```python
    result = []
    k = 10
    theshold = 0.3
    with open(file_path, "r") as file: 
        s = file.read()
        #####
        if model == "ft":
            cat_predicted = ft_model.predict(s.replace("\n", ""), k=k,threshold=threshold)
            for entry in cat_predicted[0]:
                result.append(int(entry[9:]))
        #####
        if model == "stm":
            cat_predicted = stm_model.predict_doc(s, top_k=k,ranking_threshold=threshold*100)
            for entry in cat_predicted[:k]:
                result.append(entry[1])
        ####
        if model == "svm":
            scores = stm_model.get_class_scores_doc(s)
            scores_scaled = svm_scaler.transform(scores.reshape(1, -1))
            res = svm_model.predict_proba(scores_scaled)
            cat_predicted = [sorted(list(zip(weight, svm_model.classes_)), reverse=True)[:k]for weight in res ]
            for entry in cat_predicted[0]:
                if entry[0] < threshold: continue
                result.append(entry[1])

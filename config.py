class paths:
    output_model_path = "./saved_data/models/"
    output_pred_path = "./saved_data/predictions/"
 
    anorexia_train = "./data/anorexia/Anorexia-train-time-series-data"
    anorexia_train_label = "./data/anorexia/anorexia_train_label.csv"

    anorexia_test = "./data/anorexia/Anorexia-test-time-series-data"
    anorexia_test_label = "./data/anorexia/anorexia_test_label.txt"

    anorexia_train_ffn = "./data/anorexia/extracted_features_train_anorexia.csv"
    # anorexia_train_ffn = "/raid/nlp/rajak/FFN_TIME_SERIES_RANDOM/extracted_features-train-anorexia5.csv"
    anorexia_train_label_ffn = "./data/anorexia/train_label_anorexia_ffn.csv"
    anorexia_test_ffn = "./data/anorexia/extracted_features_test_anorexia.csv"
    anorexia_corr_rank = "./data/anorexia/anorexia_30_features_correlation_ranking.csv"

    depression_train_ffn = "./data/depression/extracted_features_train_depression.csv"
    # depression_train_ffn = "/raid/nlp/rajak/FFN_TIME_SERIES_RANDOM/extracted_features-train-depression5.csv"
    depression_train_label_ffn = "./data/depression/train_label_depression_ffn.csv"
    depression_test_ffn = "./data/depression/extracted_features_test_depression.csv"
    depression_corr_rank = "./data/depression/depression_30_features_correlation_ranking.csv"

    self_harm_train_ffn = "./data/self-harm/extracted_features_train_self_harm.csv"
    # self_harm_train_ffn = "/raid/nlp/rajak/FFN_TIME_SERIES_RANDOM/extracted_features-train-selfharm5.csv"
    self_harm_train_label_ffn = "./data/self-harm/train_label_self_harm_ffn.csv"
    self_harm_test_ffn = "./data/self-harm/extracted_features_test_self_harm.csv"
    self_harm_corr_rank = "./data/self-harm/self_harm_30_features_correlation_ranking.csv"

    depression_train = "./data/depression/Depression-train-time-series-data"
    depression_train_label = "./data/depression/Depression_train_label.csv"

    depression_test = "./data/depression/Depression-test-time-series-data"
    depression_test_label = "./data/depression/Depression_test_label.txt"
 

    self_harm_train = "./data/self-harm/Self-Harm-train-time-series-data"
    self_harm_train_label = "./data/self-harm/Self-Harm_train_label.csv"

    self_harm_test = "./data/self-harm/Self-Harm-test-time-series-data"
    self_harm_test_label = "./data/self-harm/Self-Harm_test_label.txt"

    anor_by_depr_train = "./data/anorexia/Anorexia-train-time-series-data-replaced-by-[depression]"
    anor_by_sh_train = "./data/anorexia/Anorexia-train-time-series-data-replaced-by-[self-harm]"
    
    depr_by_anor_train = "./data/depression/Depression-train-time-series-data-replaced-by-[anorexia]"
    depr_by_sh_train = "./data/depression/Depression-train-time-series-data-replaced-by-[self-harm]"

    sh_by_anor_train = "./data/self-harm/Self-Harm-train-time-series-data-replaced-by-[anorexia]"
    sh_by_depr_train = "./data/self-harm/Self-Harm-train-time-series-data-replaced-by-[depression]"

    

    path_dict = {
        "anorexia" : {
            "train": anorexia_train, 
            "train_label": anorexia_train_label, 
    
            "test": anorexia_test, 
            "test_label": anorexia_test_label, 
            
        },
        
        "depression": {
            "train": depression_train, 
            "train_label": depression_train_label, 
    
            "test": depression_test, 
            "test_label": depression_test_label, 
            
        },
        
        "selfharm": {
            "train": self_harm_train, 
            "train_label": self_harm_train_label, 
    
            "test": self_harm_test, 
            "test_label": self_harm_test_label, 
            
        },
        
        "anorexia_by_depression": {
            "train": anor_by_depr_train, 
            "train_label": depression_train_label, 
    
            "test": anorexia_test, 
            "test_label": anorexia_test_label, 
            
        },
        
        "anorexia_by_selfharm": {
            "train": anor_by_sh_train, 
            "train_label": self_harm_train_label, 
    
            "test": anorexia_test, 
            "test_label": anorexia_test_label, 
            
        },
        
        "depression_by_anorexia": {
            "train": depr_by_anor_train, 
            "train_label": anorexia_train_label, 
    
            "test": depression_test, 
            "test_label": depression_test_label, 
            
        },
        
        "depression_by_selfharm": {
            "train": depr_by_sh_train, 
            "train_label": self_harm_train_label, 
    
            "test": depression_test, 
            "test_label": depression_test_label, 
            
        },
        
        "selfharm_by_anorexia": {
            "train": sh_by_anor_train, 
            "train_label": anorexia_train_label, 
    
            "test": self_harm_test, 
            "test_label": self_harm_test_label, 
            
        },
        
        "selfharm_by_depression": {
            "train": sh_by_depr_train, 
            "train_label": depression_train_label, 
    
            "test": self_harm_test, 
            "test_label": self_harm_test_label, 
            
        }
        
    }


    path_dict_ffn = {
        "anorexia" : {
            "train": anorexia_train_ffn, 
            "train_label": anorexia_train_label_ffn,
            "feature_rank": anorexia_corr_rank,
    
            "test": anorexia_test_ffn, 
            "test_label": anorexia_test_label, 
            
        },
        
        "depression": {
            "train": depression_train_ffn, 
            "train_label": depression_train_label_ffn, 
            "feature_rank": depression_corr_rank,
    
            "test": depression_test_ffn, 
            "test_label": depression_test_label, 
            
        },
        
        "selfharm": {
            "train": self_harm_train_ffn, 
            "train_label": self_harm_train_label_ffn, 
            "feature_rank": self_harm_corr_rank,
    
            "test": self_harm_test_ffn, 
            "test_label": self_harm_test_label, 
            
        }
    }


    path_dict_ml = {
        "anorexia" : {
            "train": anorexia_train_ffn, 
            "train_label": anorexia_train_label_ffn,
            "feature_rank": anorexia_corr_rank,
    
            "test": anorexia_test_ffn, 
            "test_label": anorexia_test_label, 
            
        },
        
        "depression": {
            "train": depression_train_ffn, 
            "train_label": depression_train_label_ffn, 
            "feature_rank": depression_corr_rank,
    
            "test": depression_test_ffn, 
            "test_label": depression_test_label, 
            
        },
        
        "selfharm": {
            "train": self_harm_train_ffn, 
            "train_label": self_harm_train_label_ffn, 
            "feature_rank": self_harm_corr_rank,
    
            "test": self_harm_test_ffn, 
            "test_label": self_harm_test_label, 
            
        }
    }


hyper_parameters= {
    "anorexia" : {
        "1dcnn":
        {
        "epochs": 50, 
        "batch_size": 16}, 
        
        "lstm":
        {
        "epochs": 50, 
        "batch_size": 8}, 
        
        "transformer":
        {
        "epochs": 50, 
        "batch_size": 16},

        "ffn":
        {
        "epochs": 200, 
        "batch_size": 16,
        "learning_rate": 1e-3,
        "correlation_threshold": 2}
        
    },
    
    "depression": {
        "1dcnn":
        {
        "epochs": 100, 
        "batch_size": 8}, 
        
        "lstm":
        {
        "epochs": 50, 
        "batch_size": 16}, 
        
        "transformer":
        {
        "epochs": 30, 
        "batch_size": 8},

        "ffn":
        {
        "epochs": 200, 
        "batch_size": 32,
        "learning_rate": 1e-4,
        "correlation_threshold": 2}
    },
    
    "selfharm": {
        "1dcnn":
        {
        "epochs": 100, 
        "batch_size": 16}, 
        
        "lstm":
        {
        "epochs": 50, 
        "batch_size": 8}, 
        
        "transformer":
        {
        "epochs": 50, 
        "batch_size": 8}, 

        "ffn":
        {
        "epochs": 200, 
        "batch_size": 2,
        "learning_rate": 1e-3,
        "correlation_threshold": 2}
        
    }
}
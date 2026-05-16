from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# import lightgbm as lgb


def anorexia_random_forest():
    model = RandomForestClassifier(random_state=42)
    return model

def anorexia_decision_tree(num_features):
    model = DecisionTreeClassifier(random_state=42)
    return model

def anorexia_adaboost(num_features):
    base_estimator = DecisionTreeClassifier(max_depth=1)
    model = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)
    return model

def anorexia_lightgbm(num_features):
    model = lgb.LGBMClassifier(random_state=42)
    return model


def anorexia_xgboost(num_features):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    return model

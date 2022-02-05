import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import math
from datetime import datetime

def addManualFeaturesList(manual_features_list_path, final_kept_features):
    if manual_features_list_path:
        list_ = pd.read_excel(manual_features_list_path)
        final_kept_features = list(list_['Variable'])
    print('There are ' + str(len(final_kept_features)) + ' added manually to run in next step.')
    return final_kept_features

class Collection:
    def __init__(self, a_list):
        self.a_list = a_list

    def show(self):
        return self.a_list

    def exclude(self, another_list):
        self.a_list = [kept_col for kept_col in self.a_list if kept_col not in another_list]
        return self

    def intersect(self, another_list):
        self.a_list = [kept_col for kept_col in self.a_list if kept_col in another_list]
        return self

    def union(self, another_list):
        self.a_list = self.a_list + [kept_col for kept_col in another_list if kept_col not in self.a_list]
        return self

def VIFResult(exogs, data):
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}
    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]
        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)
        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif
        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance
    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})
    return df_vif

def getLeftValues(array_2d):
    left_values = [pair[0] for pair in array_2d]
    return left_values

def getRightValues(array_2d):
    right_values = [pair[1] for pair in array_2d]
    return right_values

def findWorstRSquaredFeature(dependent_col, final_kept_features, df_train):
    # based on pseudo R-squared
    df_train['Intercept'] = 1
    r_value_root = 1
    worst_feature = ''
    for col in final_kept_features:
        cols = ['Intercept'] + [col]
        model = sm.Logit(df_train[dependent_col], df_train[cols])
        result = model.fit()
        r_value = result.prsquared
        llr_value = result.llr_pvalue
        if llr_value < 0.05 and r_value < r_value_root:
            r_value_root = r_value
            worst_feature = col
    return worst_feature, r_value_root

def findHighestBetaFeature(result_params):
    beta_pairs = list(result_params.reset_index().values)
    highest_beta = 0
    highest_beta_feature = None
    for pair in beta_pairs:
        if pair[1] > highest_beta:
            highest_beta_feature = pair[0]
            highest_beta = pair[1]
    return highest_beta_feature, highest_beta

def findHighestPvalueFeature(result_pvalues):
    result_list = list(result_pvalues.reset_index().values)
    highest_pvalue = 0
    highest_pvalue_feature = None
    for result in result_list:
        if result[1] > highest_pvalue:
            highest_pvalue_feature = result[0]
            highest_pvalue = result[1]
    return highest_pvalue_feature, highest_pvalue

def orderFeaturesByPvalueAsc(dependent_col, final_kept_features, df_train):
    pairs = []
    for feature in final_kept_features:
        result = LogisticModel(dependent_col, [feature], df_train).Logit()
        result_values = result.pvalues
        p_value = result_values[result_values.index == feature].values[0]
        pairs.append([feature, p_value])
    temp_df = pd.DataFrame(pairs, columns = ['features', 'p_value'])
    temp_df = temp_df.sort_values(by = 'p_value', ascending = True)
    ordered_features = list(temp_df['features'])
    return ordered_features

def findHighestVIFFeature(final_kept_features, df_train):
    df_vif = VIFResult(final_kept_features, df_train)
    vif_pairs = list(df_vif['VIF'].reset_index().values)
    highest_vif_value = 0
    highest_vif_feature = None
    for vif_pair in vif_pairs:
        if vif_pair[1] > highest_vif_value:
            highest_vif_value = vif_pair[1]
            highest_vif_feature = vif_pair[0]
    return highest_vif_feature, highest_vif_value

def generateFeaturesString(features_list):
    string = ''
    for feature in features_list:
        string += feature + ', '
    string = string[:-2]
    return string

class LogisticModel:
    def __init__(self, dependent_col, final_kept_features, df_train):
        self.dependent_col = dependent_col
        self.final_kept_features = final_kept_features
        self.df_train = df_train

    def Logit(self):
        self.df_train['Intercept'] = 1
        model = sm.Logit(self.df_train[self.dependent_col], self.df_train[['Intercept'] + self.final_kept_features])
        result = model.fit()
        return result

    def logisticSklearn(self):
        X_train = self.df_train[self.final_kept_features]
        y_train = getLeftValues(self.df_train[self.dependent_col].values)
        log_reg = LogisticRegression(C = 1e8, max_iter = 400)
        log_reg.fit(X_train, y_train)
        return log_reg

    def getAUCScore(self, df_test):
        log_reg = self.logisticSklearn()
        X_test = df_test[self.final_kept_features]
        y_test = getLeftValues(df_test[self.dependent_col].values)
        y_test_scores = log_reg.decision_function(X_test)
        auc_score = roc_auc_score(y_test, y_test_scores)
        return auc_score

    def getPDPredicted(self, df_test):
        log_reg = self.logisticSklearn()
        classes = log_reg.classes_
        X_test = df_test[self.final_kept_features]
        predicted_probs = log_reg.predict_proba(X_test)
        predicted_probs_dict = {}
        predicted_probs_dict[classes[0]] = getLeftValues(predicted_probs)
        predicted_probs_dict[classes[1]] = getRightValues(predicted_probs)
        return predicted_probs_dict

    def getPDWorstGroup(self, df_test, n_splits):
        predicted_probs_dict = self.getPDPredicted(df_test)
        bad_probs = predicted_probs_dict[1]
        avg_elems_each_split = math.ceil(len(bad_probs) / n_splits)
        df_test['Default Prob'] = bad_probs
        df_test_sorted = df_test[['GB_NEXT24M', 'Default Prob']].sort_values(by='Default Prob', ascending=False)
        worst_group = list(df_test_sorted['GB_NEXT24M'])[:avg_elems_each_split]
        pd_worst_group = sum(worst_group)/len(worst_group)
        return pd_worst_group

    def saveToExcel(self, df_test, destination_path):
        result = self.Logit()
        params_df = pd.DataFrame(result.params.reset_index().values, columns = ['Variable', 'coef_'])
        pvalues_df = pd.DataFrame(result.pvalues.reset_index().values, columns=['Variable', 'p_value'])
        vif_df = pd.DataFrame(VIFResult(self.final_kept_features, self.df_train)[['VIF']].reset_index().values, columns=['Variable', 'vif'])
        combined_df = params_df.merge(pvalues_df, how='left', on='Variable').merge(vif_df, how='left', on='Variable')[['Variable', 'coef_', 'p_value', 'vif']]
        # Performance indicators
        prsquared = result.prsquared  # Pseudo R-square
        llr_pvalue = result.llr_pvalue  # LLR p-value
        llr = result.llr  # Log-likelihood Ratio
        auc_score_train = self.getAUCScore(self.df_train)  # AUC score train
        gini_train = 2 * auc_score_train - 1  # Gini train
        auc_score_test = self.getAUCScore(df_test)  # AUC score test
        gini_test = 2 * auc_score_test - 1  # Gini test
        bad_list_train = list(self.df_train['GB_NEXT24M'])
        pd_portfolio_train = sum(bad_list_train) / len(bad_list_train)
        bad_list_test = list(df_test['GB_NEXT24M'])
        pd_portfolio_test = sum(bad_list_test) / len(bad_list_test)
        lift_train = self.getPDWorstGroup(self.df_train, 10) / pd_portfolio_train  # Lift train
        lift_test = self.getPDWorstGroup(df_test, 10) / pd_portfolio_test  # Lift test
        model_performance_df = pd.DataFrame(np.array([[prsquared, llr_pvalue, llr, auc_score_train, auc_score_test, gini_train, gini_test, lift_train, lift_test]])
                                            , columns = ['pseudo_r2_squared', 'llr_p_value', 'log_likelihood_ratio'
                                            , 'auc_score_train', 'auc_score_test', 'gini_train', 'gini_test', 'lift_train',	'lift_test'])
        predicted_probs_train_dict = self.getPDPredicted(self.df_train)
        predicted_probs_train_dict['APP_ID'] = list(self.df_train['APP_ID'])
        predicted_probs_train_dict[self.dependent_col[0]] = list(self.df_train[self.dependent_col[0]])
        pd_predicted_train_df = pd.DataFrame(predicted_probs_train_dict)
        predicted_probs_test_dict = self.getPDPredicted(df_test)
        predicted_probs_test_dict['APP_ID'] = list(df_test['APP_ID'])
        predicted_probs_test_dict[self.dependent_col[0]] = list(df_test[self.dependent_col[0]])
        pd_predicted_test_df = pd.DataFrame(predicted_probs_test_dict)
        opened_Excel = pd.ExcelWriter(destination_path)
        combined_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='coef_', index=False)
        model_performance_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='model_performance', index=False)
        pd_predicted_train_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='pd_predicted_train', index=False)
        pd_predicted_test_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='pd_predicted_test', index=False)
        opened_Excel.save()

class FeaturesElimination:
    def __init__(self, dependent_col, final_kept_features, df_train, removed_features={}):
        self.dependent_col = dependent_col
        self.final_kept_features = final_kept_features
        self.df_train = df_train
        self.removed_features = removed_features

    def backwardElimination(self, pvalue_threshold):
        begin_time = datetime.now()
        result = LogisticModel(self.dependent_col, self.final_kept_features, self.df_train).Logit()
        result_values = result.pvalues
        highest_pvalue_feature, highest_pvalue = findHighestPvalueFeature(result_values)
        if highest_pvalue > pvalue_threshold:
            self.removed_features[highest_pvalue_feature] = self.final_kept_features
            self.final_kept_features = Collection(self.final_kept_features).exclude([highest_pvalue_feature]).show()
            self.backwardElimination(pvalue_threshold)
        else:
            end_time = datetime.now()
            complete_time = (end_time - begin_time).total_seconds()
            print('Backward Elimination was completed.')
            print('Duration: ' + str(complete_time) + 's.')
        return self

    def forwardElimination(self, pvalue_threshold):
        begin_time = datetime.now()
        ordered_features = orderFeaturesByPvalueAsc(self.dependent_col, self.final_kept_features, self.df_train)
        final_features = []
        for feature in ordered_features:
            final_features.append(feature)
            result = LogisticModel(self.dependent_col, final_features, self.df_train).Logit()
            result_values = result.pvalues
            if len(result_values[result_values > pvalue_threshold]) > 0:
                print(feature)
                self.removed_features[feature] = final_features
                final_features = Collection(final_features).exclude([feature]).show()
        self.final_kept_features = final_features
        end_time = datetime.now()
        complete_time = (end_time - begin_time).total_seconds()
        print('Forward Elimination was completed.')
        print('Duration: ' + str(complete_time) + 's.')
        return self

    def vifElimination(self, vif_threshold):
        highest_vif_feature, highest_vif_value = findHighestVIFFeature(self.final_kept_features, self.df_train)
        if highest_vif_value > vif_threshold:
            self.removed_features[highest_vif_feature] = self.final_kept_features
            self.final_kept_features = Collection(self.final_kept_features).exclude([highest_vif_feature]).show()
            self.vifElimination(vif_threshold)
        return self

    def betaElimination(self, beta_threshold):
        result = LogisticModel(self.dependent_col, self.final_kept_features, self.df_train).Logit()
        result_params = result.params
        highest_beta_feature, highest_beta_value = findHighestBetaFeature(result_params)
        if highest_beta_value > beta_threshold:
            self.removed_features[highest_beta_feature] = self.final_kept_features
            self.final_kept_features = Collection(self.final_kept_features).exclude([highest_beta_feature]).show()
            self.betaElimination(beta_threshold)
        return self

    def StepsBaseElimination(self, pvalue_threshold, vif_threshold, beta_threshold, option):
        if option == 'backward':
            self.backwardElimination(pvalue_threshold)
        if option == 'forward':
            self.forwardElimination(pvalue_threshold)
        if option == 'both':
            self.backwardElimination(pvalue_threshold)
            self.forwardElimination(pvalue_threshold)
        self.betaElimination(beta_threshold)
        self.vifElimination(vif_threshold)
        result = LogisticModel(self.dependent_col, self.final_kept_features, self.df_train).Logit()
        result_pvalues = result.pvalues
        result_params = result.params
        _, highest_p_value = findHighestPvalueFeature(result_pvalues)
        _, highest_beta = findHighestBetaFeature(result_params)
        _, highest_vif = findHighestVIFFeature(self.final_kept_features, self.df_train)
        if highest_p_value > pvalue_threshold or highest_vif > vif_threshold or highest_beta > beta_threshold:
            self.StepsBaseElimination(pvalue_threshold, vif_threshold, beta_threshold, option)
        return self

    def excludeFeaturesToMaxNumber(self, pvalue_threshold, vif_threshold, beta_threshold, max_number, option):
        self.StepsBaseElimination(pvalue_threshold, vif_threshold, beta_threshold, option)
        if len(self.final_kept_features) > max_number:
            worst_feature, _ = findWorstRSquaredFeature(self.dependent_col, self.final_kept_features, self.df_train)
            self.removed_features[worst_feature] = self.final_kept_features
            self.final_kept_features = Collection(self.final_kept_features).exclude([worst_feature]).show()
            self.excludeFeaturesToMaxNumber(pvalue_threshold, vif_threshold, beta_threshold, max_number)
        return self

    def saveToExcel(self, destination_path):
        kept_features_df = pd.DataFrame(np.array(self.final_kept_features).T, columns = ['kept_features'])
        excluded_features = []
        senarios = []
        for feature in list(self.removed_features.keys()):
            features_string = generateFeaturesString(self.removed_features[feature])
            excluded_features.append(feature)
            senarios.append(features_string)
        excluded_features_df = pd.DataFrame(np.array([excluded_features, senarios]).T, columns = ['removed_features', 'scenarios'])
        opened_Excel = pd.ExcelWriter(destination_path)
        kept_features_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='kept_features', index=False)
        excluded_features_df.to_excel(opened_Excel, engine='xlsxwriter', sheet_name='removed_features', index=False)
        opened_Excel.save()
        return self.final_kept_features


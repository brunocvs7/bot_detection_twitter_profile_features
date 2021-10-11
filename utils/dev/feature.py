from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer 
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np


def point_biserial(df, y, num_columns = None, significance=0.05):
    '''
    Perform feature selection based on correlation test.

            Parameters:
                    df (pandas.dataframe): A dataframe containing all features and target
                    num_columns (list): A list containing all categorical features. If empty list, the function tries to infer the categorical columns itself
                    y (string): A string indicating the target.

            Returns:
                    columns_remove_pb (list): 

    '''
    correlation = []
    p_values = []
    results = []
    
    
    if num_columns:
        num_columns = num_columns
    else:
        num_columns = df.select_dtypes(include=['int','float', 'int32', 'float64']).columns.tolist()
    
    
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
        correlation_aux, p_value_aux = pointbiserialr(df[col], df[y])
        correlation.append(correlation_aux)
        p_values.append(p_value_aux)
    
    
        if p_value_aux <= significance:
            results.append('Reject H0')
        else:
            results.append('Accept H0')
    
    
    pb_df = pd.DataFrame({'column':num_columns, 'correlation':correlation, 'p_value':p_values, 'result':results})
    columns_remove_pb =  pb_df.loc[pb_df['result']=='Accept H0']['column'].values.tolist()
  
    
    return pb_df, columns_remove_pb


class Boruta:
    """
    A class to perform feature selection, based on BorutaPy Class of boruta package
    This version is based only on feature importance of a random forest model and returns results more pretifully
    See https://github.com/scikit-learn-contrib/boruta_py for more details (original implementation)

    ...

    Attributes
    ----------
    n_iter : int
        number of iterations the algorithm will perform
    columns_removed : list 
        list of columns to be removed (Obtained after fit method runs)
 

    Methods
    -------
    fit(X, y):
        Runs Boruta Algorithm. It brings a list of columns We should remove and a boolean vetor.
    """

    def __init__(self, n_iter=100):
        """
        Constructs all the necessary attributes for the boruta object.

        Parameters
        ----------
        n_iter : int
            number of iterations the algorithm will perform
        """
        self.n_iter = n_iter
        self._columns_remove_boruta = None
        self._bool_decision = None
        self._best_features = None

    def fit(self, X, y, cat_columns=True, num_columns=True):
        """
        Runs Boruta Algorithm.

        Parameters
        ----------
        X : pandas.dataframe
            Pandas Data Frame with all features
        y: pandas.dataframe
            Pandas Data Frame with target
    
        Returns
        -------
        None
        """
        X.replace(to_replace=[None], value=np.nan, inplace=True)
        if (num_columns == False) & (cat_columns == True):
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            X.loc[:, cat_columns] = X.loc[:, cat_columns].astype('str')
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X,y)
        elif (cat_columns==False) &  (num_columns==True):
            num_columns = X.select_dtypes(include=['int','float']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        else:     
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            X.loc[:, cat_columns] = X.loc[:, cat_columns].astype('str')
            num_columns = X.select_dtypes(include=['int','float']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        bool_decision = [not x for x in selector.support_.tolist()] # apenas invertendo o vetor de true/false
        columns_remove_boruta = X.loc[:,bool_decision].columns.tolist()
        columns_keep_boruta = X.loc[:,selector.support_.tolist()].columns.tolist()
        self._columns_remove_boruta = columns_remove_boruta
        self._bool_decision = bool_decision
        self._best_features = columns_keep_boruta

def chi_squared(df, y, cat_columns = None, significance=0.05):
    '''
    Performs chi2 hypothesis test to find relationship between predictors and target in a data frame

            Parameters:
                    df (pandas.dataframe): A data frame containing categorical features and target variable
                    y (string): A string that saves the name of target variable
                    cat_columns (list): A list with the name of categorical features. If None, function tries to infer It by itself
                    significance (float): A float number indicating the significance level for the test. Deafult is 0.05

            Retorna:
                    chi2_df (pandas.dataframe): A data frame with the results of the tests
                    columns_remove_chi2 (list): A list of columns that should be removed
                    logs (list): A list of columns that could not be evaluated
    '''
    
    
    p_values = []
    logs = []
    chi2_results = []
    results = []
    
    
    if cat_columns == None:
        cat_columns = df.select_dtypes(['object']).columns.tolist()
    else:
        cat_columns = cat_columns
        
        
    for cat in cat_columns:    
        cross_table = pd.crosstab(df[cat], df[y])
        
        
        if not cross_table[cross_table < 5 ].count().any():    
            cross_table = pd.crosstab(df[cat], df[y])
            chi2, p, dof, expected = chi2_contingency(cross_table.values)
            chi2_results.append(chi2)
            p_values.append(p)
        else:
            logs.append("Column {} could'nt be evaluated".format(cat))
            chi2_results.append(np.nan)
            p_values.append(np.nan)
            
            
    for p in p_values:
        
        
        if p <= significance:
            results.append('Reject H0')
        else:
            results.append('Accept H0')   
            
            
    chi2_df = pd.DataFrame({"column":cat_columns, 'p-value':p_values,'chi2':chi2_results, 'results':results})
    columns_remove_chi2 =  chi2_df.loc[chi2_df['results']=='Accept H0']['column'].values.tolist()
    return  chi2_df, columns_remove_chi2, logs
"""
Provides standard complexity measures and related descriptive statistics.
"""
import pandas as pd
import numpy as np
import itertools
import math
import random
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def f1_maximum_fisher_discriminating_ratio ( data, result = True):
    """
    Calculates de f1 statistic from a dataset
    @params:
        data    - Required: a pandas DataFrame with class labels on last column
        result  - Optional: returns only f1 value (for True) or f1 and feature (False) (Int or Boolean)
    @output:
        r_{f_i} = \frac{\sum_{c_j=1}^{k} \sum_{c_l=c_j+1}^{k} p_{c_j} p_{c_l} (\mu_{c_j}^{f_i} - \mu_{c_l}^{f_i})^2}{\sum_{c_j=1}^{k} p_{c_j} \sigma_{c_j}^{2}}
    """
    classes = data.iloc[:,-1].unique()
    features = data.columns[:-1,]
    n = data.shape[0]
    F1 = 0
    
    def f(x, y):
        return pow(x - y, 2)
    
    for feat in features:
        rfi = 0
        for ci in classes:
            nci = data[data.iloc[:,-1] == ci].shape[0]
            mi_ci  = data[data.iloc[:,-1] == ci][feat].mean()
            feat_mean = data[feat].mean()
            sq_error = 0
            for item in data[data.iloc[:,-1] == ci][feat]:
                sq_error = sq_error + f(item, feat_mean)
                
            rfi = rfi + ((1.0 * nci * f(mi_ci, feat_mean)) / sq_error )
        if F1 < rfi:
            max_feat = feat
            F1 = rfi   
    if result:
        return F1
    else:
        return (F1, max_feat)

def get_minmax_maxmin(c1_data, c2_data, feat):
        
    minmax_fi = min(c1_data[feat].max(),
                    c2_data[feat].max())
    maxmin_fi = max(c1_data[feat].min(),
                    c2_data[feat].min())
    return minmax_fi, maxmin_fi

def f2_volume_of_overlapping(data):
    """
    Calculates de f1 statistic from a dataset
    @params:
        data    - Required: a pandas DataFrame with class labels on last column
        result  - Optional: returns only f1 value (for True) or f1 and feature (False) (Int or Boolean)
    @output:
        
    """           
    def get_f2_for_tupla(data, c1, c2, feat):
        c1_data = data[data.iloc[:,-1] == c1]
        c2_data = data[data.iloc[:,-1] == c2]
        
        minmax_fi, maxmin_fi = \
        get_minmax_maxmin(c1_data, c2_data, feat)
        maxmax_fi = max(c1_data[ feat ].max(),
                        c2_data[ feat ].max())
        minmin_fi = min(c1_data[ feat ].min(),
                        c2_data[ feat ].min())
        overlap = max(0 , minmax_fi - maxmin_fi)
        rang = maxmax_fi - minmin_fi
        if rang == 0:
            return 1
        return ((1.0 * overlap) / rang)
        
    classes = data.iloc[:,-1].unique()
    features = data.columns[:-1,]
    f2_cl = 1
    f2 = 0
    
    for subset in itertools.combinations(classes, r=2):
        for feat in features:
            c1 = subset[0]
            c2 = subset[1]
            f2_cl = f2_cl * get_f2_for_tupla(data, c1, c2, feat)
        f2 = f2 + f2_cl
        
    return np.mean(f2_cl)

def get_f3_for_tupla(data, c1, c2, features):

    c1_data = data[data.iloc[:,-1] == c1]
    c2_data = data[data.iloc[:,-1] == c2]
    
    data_i = data[(data.iloc[:, -1] == c2) | (data.iloc[:, -1] == c1)]
    data_i.reset_index(drop=True, inplace = True)
    
    f3_i = 0 # number of examples that are not in the overlapping region
    max_feat = features[0]
    
    for feat in features:
        minmax_fi, maxmin_fi = \
        get_minmax_maxmin(c1_data, c2_data, feat)
        n_fi = len(data_i[(data_i[feat] > minmax_fi) | (data_i[feat] < maxmin_fi)])
        
        if n_fi > f3_i:
            max_feat = feat
            f3_i = n_fi
            
    n = data_i.shape[0] # total number of examples
    f3_i = (1.0 * f3_i) / n

    return f3_i, max_feat
        
def f3_maximum_individual_feat_efficiency(data, result = True):    

    classes = data.iloc[:,-1].unique()
    features = data.columns[:-1, ] # list of features
    F3 = []
    max_feat = features[0]
    
    for subset in itertools.combinations(classes, r=2):
        c1 = subset[0]
        c2 = subset[1]
        f3_i, _ = get_f3_for_tupla(data, c1, c2, features)
        
        F3.append(f3_i)
    
    return np.mean(F3)

def f4_collective_reature_efficiency(data):

    def items_in_overlap(feat, data, items_in_overlap, minmax_fi, maxmin_fi):
        items_in_overlap = [not(a or b) and c
                            for a, b, c in zip(
                                data[feat] > minmax_fi, 
                                data[feat] < maxmin_fi,
                                items_in_overlap)]
        return items_in_overlap
    
    def get_f4_for_tupla(data, c1, c2):
        c1_data = data[data.iloc[:,-1] == c1]
        c1_data.reset_index(drop=True, inplace = True)
        c2_data = data[data.iloc[:,-1] == c2]
        c2_data.reset_index(drop=True, inplace = True)
        
        data_i = data[(data.iloc[:, -1] == c2) | (data.iloc[:, -1] == c1)]
        data_i.reset_index(drop=True, inplace = True)
        
        n_features = len(data.columns[:-1, ])
        features_i = list(data.columns[:-1]) # a copy for modification
        
        nc1 = c1_data.shape[0]
        c1_items_in_overlap = [True] * nc1
        nc2 = c2_data.shape[0]
        c2_items_in_overlap = [True] * nc2
        
        for idx in range(n_features):

            _, f3_feat = get_f3_for_tupla(data_i, c1, c2, features_i)
            features_i.remove(f3_feat)
            
            c1_data = c1_data.iloc[c1_items_in_overlap, :]
            c2_data = c2_data.iloc[c2_items_in_overlap, :]
            
            minmax_fi, maxmin_fi = \
            get_minmax_maxmin(c1_data, c2_data, f3_feat)
                    
            c1_items_in_overlap = \
            items_in_overlap(f3_feat, c1_data, c1_items_in_overlap,
            minmax_fi, maxmin_fi)
 
            c2_items_in_overlap = \
            items_in_overlap(f3_feat, c2_data, c2_items_in_overlap,
            minmax_fi, maxmin_fi)
            
            if not (any(c1_items_in_overlap) or any(c2_items_in_overlap)):
                break

        n = data_i.shape[0]
        n_items_off_overlap = n - sum(c1_items_in_overlap) - \
            sum(c2_items_in_overlap)
        f4_i = (1.0 * n_items_off_overlap) /  n
        
        return f4_i
        
    classes = data.iloc[:,-1].unique()
    F4 = []
    
    for subset in itertools.combinations(classes, r = 2):
        c1 = subset[0]
        c2 = subset[1]
        
        f4_i = get_f4_for_tupla(data, c1, c2)
        F4.append(f4_i)
        
    return np.mean(F4)
        

def l1_sum_of_error_distance(data):

    def get_l1_for_tupla(data, c1, c2):
        
        features = data.columns[:-1, ]
        labels   = data.columns[-1]
        
        c1_indexes = data[data.iloc[:, -1] == c1].index.values
        c2_indexes = data[data.iloc[:, -1] == c2].index.values
        indexes    = np.concatenate((c1_indexes, c2_indexes), axis=0)

        X = data[features].iloc[indexes, :]
        y = data[labels][indexes].tolist()
        
        
        # forcando as classes serem 1 ou -1.
        for i in range(len(y)):
            if y[i] == c1:
                y[i] = -1
            else: y[i] = 1
            
        vec = y.copy()

        clf = svm.LinearSVC(loss='hinge')
        clf.fit(X, y)
        l1 = 0
        regression = 0
        
        # verificar essa parte de classes internas do SVM
        
        def f(x, y):
            return pow(x - y, 2)

        #c1 = 1 if regression > 0 else -1
        
        # Calculando a diagonal do hyperretangulo
        min_hyper = []
        max_hyper = []
        for i in X.columns:
            min_hyper.append(X[i].unique().min())
            max_hyper.append(X[i].unique().max())

        dist_hyper = np.sqrt(np.sum(f(np.asarray(min_hyper), np.asarray(max_hyper))))
        if dist_hyper == 0:
            dist_hyper = 1
        
        #dist_max = 1    
        for idx, row in enumerate(X.iterrows()):

            temp = np.array(row[1]).reshape(1,-1) 
            regression = clf.decision_function(temp) # temp é um exemplo
            
            val = y[idx] * regression
            
            if(val < 0):
                l1 = l1 + (1 - val)

        return (1.0 * l1) / (n * dist_hyper)
        
    classes = data.iloc[:, -1].unique()
    n  = data.shape[0]
    L1 = []
       
    for subset in itertools.combinations(classes, r=2):
        c1 = subset[0]
        c2 = subset[1]
        
        l1_ci = get_l1_for_tupla(data, c1, c2)
        
        L1.append(l1_ci)
        
    return float(np.mean(L1))
    
def l2_rate_of_linear_classifier(data):
    
    def get_l2_for_tupla(data, c1, c2):
        features = data.columns[:-1, ]
        labels = data.columns[-1]
        clf = svm.LinearSVC(loss='hinge')
        c1_indexes = data[data.iloc[:, -1] == c1].index.values
        c2_indexes = data[data.iloc[:, -1] == c2].index.values
        indexes = np.concatenate((c1_indexes, c2_indexes), axis=0)

        X = data[features].iloc[indexes, :]
        y = data[labels][indexes]
        
        clf.fit(X, y)
        for i in range(X.shape[0]):
            temp = np.array(data.iloc[i ,:-1]).reshape(1,-1)
        return 1.0 - clf.score(X, y)
        
    labels = data.columns[-1]
    classes = data[labels].unique()
    L2 = []
    
    for subset in itertools.combinations(classes, r=2):
        c1 = subset[0]
        c2 = subset[1]
        
        l2_ci = get_l2_for_tupla(data, c1, c2)
        L2.append(l2_ci)
    return np.mean(L2)
    
    
def n1_fraction_borderline(data):

    def get_n1_for_round(sparse_matrix, y):
        Tcsr = minimum_spanning_tree(sparse_matrix)
        borders = set()
        a = Tcsr.nonzero()[0]
        b = Tcsr.nonzero()[1]
        
        for i in range(len(a)):
            if (y[a[i]] != y[b[i]]):
                borders.add(a[i])
                borders.add(b[i])
        n1 = len(borders)
        return n1
        
    features = data.columns[:-1, ]
    dist = pdist(data[features], 'euclidean')
    df_dist = pd.DataFrame(squareform(dist))
    sparse_matrix = csr_matrix(df_dist.values)
    
    labels = data.columns[-1]
    y = data[labels]
    
    n1 = 0
    rounds = 10
    
    for round in range(rounds):
        n1 = n1 + get_n1_for_round(sparse_matrix, y)
                
    n = len(data)
    n1 = (1.0 * n1) / (rounds * n)
    
    return n1
    
    
def n2_ratio_intra_extra_class_nearest_neighbor_distance(data):
       
    features = data.columns[:-1,]
    labels = data.columns[-1]
    
    dist    = pdist(data[features], 'euclidean')
    df_dist = pd.DataFrame(squareform(dist))

    max_size = df_dist.copy( )
    max_size.iloc[:, :] = False
    
    classes = data.iloc[ :, -1].unique()
    n = data.shape[0]
    
    n2 = 0
    cl = 'bla'
    intra_min = 0
    inter_min = 0
    for i in range(data.shape[0]):
        ci = data.iloc[i, -1]
        if ci != cl:
            cl = ci
            intra_idx = data[data[labels] == ci].index.values.tolist()
            inter_idx = data[data[labels] != ci].index.values
        intra_idx.remove(i)
        intra_min = intra_min + df_dist.iloc[intra_idx, i].min()
        inter_min = inter_min + df_dist.iloc[inter_idx, i].min()
        intra_idx.append(i)
        
    # tratar caso de inter_min == 0
    if inter_min == 0:
        inter_min = 1

    n2 = (1.0 * intra_min) / (1.0 * inter_min)
        
    return n2


def n3_error_rate_nearest_neighbor_classifier(data):

    features = data.columns[:-1, ]
    mistakes = 0
    n = data.shape[0]
    
    for i in range(n):
        bad_df = data.index.isin([i])
        good_df = ~bad_df
        
        knn = KNeighborsClassifier( n_neighbors=1 )
        knn.fit(data.iloc[good_df].iloc[:, :-1], data.iloc[good_df].iloc[: ,-1])
        temp = np.array(data.iloc[i ,:-1]).reshape(1,-1)
        mistake = 1 if data.iloc[i, -1] != knn.predict(temp) else 0
        
        mistakes = mistakes + mistake
    
    n3 = (1.0 * mistakes) / n
    if n3 > 1:
        n3 = 1
    return n3

def random_combinations(points_in_class):
    n_cl = len(points_in_class)
    max_points = 2 * n_cl # as used by by Orriols-Puig et al., 2010
    all_combinations = []
    for i, j in itertools.combinations(points_in_class, r = 2):
        all_combinations.append((i, j))
    
    points_i = 0
    n = len(all_combinations)
    for i in range(n):
    
        point = np.random.choice(len(all_combinations), 1)[0]

        yield all_combinations[point]
        
        del all_combinations[point]
        if points_i > max_points or len(all_combinations) == 0:
            break
        points_i = points_i + 1

def linear_interpolation(a, b):
    alpha = random.random()
    result = (1 - alpha)*a + alpha*b
    return result.tolist()

def l3_non_linearity_of_linear_classifier(data, random_seed = 42, iterations = 20):
        
    def get_score_interpolation(X, y, iterations = iterations):
        clf = svm.LinearSVC(loss='hinge')
        clf.fit(X, y)
        score = 0
        
        for i in range(iterations):
            X_interpolated = []
            y_interpolated = []
            
            for cl in subset:
                points_in_class = y[y == cl].index.tolist()
                n_cl = len(points_in_class)

                it_points = 0
                
                for a, b in random_combinations(points_in_class):
                    new_point = linear_interpolation(X.iloc[a, :], X.iloc[b, :])
                    X_interpolated.append(new_point)
                    y_interpolated.append(cl)
                    
            score = score + (1.0 - clf.score(X_interpolated, y_interpolated))
          
        return score
        
    def get_l3_for_tupla(data, c1, c2, iterations = iterations):
        data_i = data[(data.iloc[:, -1] == c2) | (data.iloc[:, -1 ] == c1)]
        data_i.reset_index(drop=True, inplace = True)
        
        X = data_i[features]
        y = data_i[labels]
        
        score = get_score_interpolation(X, y)
        
        return (1.0 * score) / iterations
        
    random.seed(random_seed)
    classes = data.iloc[:,-1].unique()
    n = data.shape[0]
    L3 = []
    features = data.columns[:-1,]
    labels = data.columns[-1]
    
    for subset in itertools.combinations(classes, r=2):
        c1 = subset[0]
        c2 = subset[1]
        
        l3 = get_l3_for_tupla(data, c1, c2)
        L3.append(l3)

    return np.mean(L3)   
    
def n4_non_linearity_of_nearest_neighbor_classifier( data, random_seed = 42, iterations = 20 ):
        
    def generate_interpolated_data_cl(data, cl, features, labels):
        points_in_class = data[data[labels] == cl].index.tolist()
        data_interpolated = pd.DataFrame(columns = features + [labels])
        
        for a, b in random_combinations(points_in_class):
            new_point = linear_interpolation(data.iloc[a, :-1], data.iloc[b, :-1] )
            df = pd.DataFrame([new_point + [cl]], columns = features + [labels] )
            data_interpolated = data_interpolated.append(df)

        return data_interpolated

    def get_n4_for_iteration(data):  
        
        labels = data.columns[-1]
        features = data.columns[:-1,].tolist()
        classes = data.iloc[:, -1].unique()
        data_to_interpolate = data.copy()
        
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(data[features], data[labels])
                
        for cl in classes:
            data_interpolated = generate_interpolated_data_cl(data_to_interpolate, cl, features, labels)
            
        mistakes = 1 - knn.score(data_interpolated[features], data_interpolated[labels])
        
        return mistakes
        
    random.seed( random_seed )
    n4 = []
    
    for i in range(iterations):
        mistakes = get_n4_for_iteration(data)
        n4.append(mistakes)
    
    return np.mean(n4)
    
    
def t1_fraction_hyperspheres_covering_data(data):

    def get_min(matrix, to_process):
        min_dist = math.inf
        for row_idx, row in matrix.iterrows():
            for col_idx, item in row.iteritems():
                if to_process.iloc[row_idx, col_idx] and \
                min_dist > matrix.iloc[row_idx, col_idx]:
                    min_dist = matrix.iloc[row_idx, col_idx]
        return min_dist
        
    def get_value_idxs(matrix, value, to_process):
        for row_idx, row in matrix.iterrows():
            for col_idx, item in row.iteritems():
                if to_process.iloc[row_idx, col_idx] and \
                value == matrix.iloc[row_idx, col_idx]:
                    yield (row_idx, col_idx)
                    
    def processa_min_dist(df_dist, row_idx, col_idx, max_size, to_process, dist):
        if max_size.iloc[:, col_idx].all() and max_size.iloc[row_idx, :].all():
            pass
        
        elif max_size.iloc[:, col_idx].all():
            df_dist.iloc[row_idx, :] =  df_dist.iloc[row_idx, :] - dist
            max_size.iloc[row_idx, :] = True
            
            ####simetria
            df_dist.iloc[:, row_idx] =  df_dist.iloc[:, row_idx] - dist
            max_size.iloc[:, row_idx] = True

        elif max_size.iloc[row_idx, :].all():
            df_dist.iloc[:, col_idx] =  df_dist.iloc[:, col_idx] - dist
            max_size.iloc[:, col_idx] = True
            
            ####simetria
            df_dist.iloc[col_idx, :] =  df_dist.iloc[col_idx, :] - dist
            max_size.iloc[col_idx, :] = True

        else:
            df_dist.iloc[row_idx, :] =  df_dist.iloc[row_idx, :] - dist/2
            df_dist.iloc[:, col_idx] =  df_dist.iloc[:, col_idx] - dist/2
            max_size.iloc[:, col_idx] = True
            max_size.iloc[row_idx, :] = True
            
            ####simetria
            df_dist.iloc[col_idx, :] =  df_dist.iloc[col_idx, :] - dist/2
            df_dist.iloc[:, row_idx] =  df_dist.iloc[:, row_idx] - dist/2
            max_size.iloc[col_idx, :] = True
            max_size.iloc[:, row_idx] = True
        
        to_process.set_value(col_idx, row_idx, False)
        ####simetria
        to_process.set_value(row_idx, col_idx, False)
        return df_dist, max_size, to_process
        
    def get_t1_for_cl(df_dist, c1_indexes, c2_indexes, max_size):
    
        to_process = df_dist.copy()
        to_process.iloc[:, :] = False
        to_process.iloc[c1_indexes, c2_indexes] = True
        
        while to_process.any().any():
            
            min_dist = get_min(df_dist, to_process)
            a = data == min_dist
            for row_idx, col_idx in get_value_idxs(df_dist, min_dist, to_process):
                df_dist, max_size, to_process = processa_min_dist(df_dist, row_idx, 
                col_idx, max_size, to_process, min_dist)
                               
        return df_dist, max_size
        
    features = data.columns[:-1]
    
    dist    = pdist(data[features], 'euclidean')
    df_dist = pd.DataFrame(squareform(dist))

    max_size = df_dist.copy( )
    max_size.iloc[:, :] = False
    
    classes = data.iloc[ :, -1].unique()
    n = data.shape[0]
    n_hyperspheres = [ ]
       
    for cl in classes[:-1]:
        cl_indexes = data[data.iloc[:, -1] == cl].index.values
        not_cl_indexes = data[data.iloc[:, -1] != cl].index.values
        
        df_dist, max_size = get_t1_for_cl(df_dist, cl_indexes, not_cl_indexes, max_size)
        
    n_hs = 0   
    for i in (df_dist < 0).sum():
        if i == 0: # como tratar i == 0:
            n_hs = n_hs + 1
        else:
            n_hs = n_hs + 1.0/i
    return round(1.0 * n_hs / n, 2)
    
    
def t2_average_number_of_examples_per_dimension(data):
    t2 = data.shape[0]/(data.shape[1] - 1)
    return t2


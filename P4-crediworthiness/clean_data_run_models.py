import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV as GSCV
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from statsmodels.tools.tools import add_constant

def load_full_data():
    '''
    loads train, test, and var_types dataframes
    '''
    var_types = pd.read_csv('variable_types.csv')
    train = pd.read_excel('credit-data-training.xlsx')
    test = pd.read_excel('customers-to-score.xlsx')
    return train, test, var_types

def examine_vars(train, var_types):
    '''
    looks at all variables in the training set to check which we should keep
    looks at histograms of numeric data, and value_counts of non-numeric

    args:
    train -- dataframe of training set
    var_types -- dataframe with columns: Variable, Data Type

    returns nothing
    '''
    print(train.info()) # Duration-in-Current-address has too many missing values

    print('\n')
    for i, v in var_types.iterrows():
        var = v['Variable']
        print(var)
        if v['Data Type'] == 'String':
            print(v, train[var].value_counts())
            print('\n')
        else:
            train[var].hist(bins=20)
            plt.xlabel(var)
            plt.show()

def clean_data(train, test, var_types):
    '''
    drops unnecessary columns, encodes string vars as numeric, and creates
    dummied dataframes for train/test
    '''
    drop_cols = ['Duration-in-Current-address', 'Type-of-apartment', 'Occupation',
                'No-of-dependents', 'Foreign-Worker', 'Concurrent-Credits',
                'Guarantors']
    for c in drop_cols:
        train.drop(c, inplace=True, axis=1)
        test.drop(c, inplace=True, axis=1)

    var_types = var_types[~var_types['Variable'].isin(drop_cols)]

    # mean-impute age in training; no ages missing in test
    train['Age-years'].fillna(int(train['Age-years'].mean()), inplace=True)
    train['Age-years'] = train['Age-years'].astype('int64')
    print(train.corr())

    # encode target as numeric for logistic regression/correlation analysis
    train['Credit-Application-Result'] = train['Credit-Application-Result']. \
                                            apply(lambda x: \
                                            1 if x == 'Creditworthy' \
                                            else 0)

    train_dummies = train.copy()
    test_dummies = test.copy()
    label_encoders = []
    for i, v in var_types[var_types['Data Type'] == 'String'].iterrows():
        var = v['Variable']
        if var == 'Credit-Application-Result':
            continue
        le = LE()
        label_encoders.append(le)
        train[var] = le.fit_transform(train[var])
        test[var] = le.transform(test[var])
        train_dummies = pd.get_dummies(train_dummies, columns=[var])
        test_dummies = pd.get_dummies(test_dummies, columns=[var])

    # found out 'Purpose_Other' wasn't in the test data
    test_dummies['Purpose_Other'] = 0
    # also found out the column order got switched
    test_dummies = test_dummies[train_dummies.columns.values[1:]]

    return train, test, train_dummies, test_dummies


def make_train_val(train):
    # make train-validation sets
    cols = list(train.columns.values)
    cols.remove('Credit-Application-Result')
    y = train['Credit-Application-Result'].values
    X = train[cols].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_val, y_train, y_val

def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    # thanks, matt: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train, test, var_types = load_full_data()
    # heatmap of training data correlation before cleaning it
    # first need to convert strings to numbers and fill NANs
    temp_train = train.copy()
    for c in temp_train.columns:
        if temp_train[c].dtype == 'O':
            le = LE()
            temp_train[c] = le.fit_transform(temp_train[c])

    temp_train['Age-years'] = temp_train['Age-years'].fillna(temp_train['Age-years'].mean())
    # too many NANs to be useful
    temp_train.drop('Duration-in-Current-address', inplace=True, axis=1)
    # all one value
    temp_train.drop('Concurrent-Credits', inplace=True, axis=1)
    temp_train.drop('Occupation', inplace=True, axis=1)

    cmap = plt.cm.jet
    cmap.set_over('g')
    cmap.set_under('g')
    sns.heatmap(temp_train.corr(),
                vmax=0.7,
                vmin=-0.7,
                square=True,
                cmap=cmap,
                xticklabels=range(temp_train.shape[1]),
                yticklabels=range(temp_train.shape[1]),
                )
    plt.show()

    # print columns for a key to heatmap:
    for i, t in enumerate(temp_train.columns.values):
        print(str(i).ljust(2), ':', t)

    train, test, train_dummies, test_dummies = clean_data(train, test, var_types)
    X_train, X_val, y_train, y_val = make_train_val(train_dummies)

    print(train_dummies.corr()) # pearson correlations
    print(train.corr())
    # correlation heatmap
    cmap = plt.cm.jet
    cmap.set_over('g')
    cmap.set_under('g')
    sns.heatmap(train.corr(),
                vmax=0.7,
                vmin=-0.7,
                square=True,
                cmap=cmap,
                xticklabels=range(train.shape[1]),
                yticklabels=range(train.shape[1]),
                )
    plt.show()

    # print columns for a key to heatmap:
    for t in train.columns.values:
        print(t)

    # logistic regression
    logit = Logit(y_train, add_constant(X_train))
    res = logit.fit()
    print(res.summary()) # looks like only var #2 is important (P < 0.05)
    print(train_dummies.columns[2]) # that's Credit-Amount
    # sns.violinplot(x='Duration-of-Credit-Month', y='Credit-Application-Result', data=train)
    # sns.violinplot(x='Instalment-per-cent', y='Credit-Application-Result', data=train, scale='count')
    sns.violinplot(x='Credit-Application-Result', y='Credit-Amount', data=train, scale='count')
    plt.show()
    # lower credit amounts are approved more often
    # next 2 most important are 'Instalment-per-cent', 'Most-valuable-available-asset'
    sns.violinplot(x='Credit-Application-Result', y='Instalment-per-cent', data=train, scale='count')
    plt.show()
    sns.violinplot(x='Credit-Application-Result', y='Most-valuable-available-asset', data=train, scale='count')
    plt.show()

    preds = res.predict(add_constant(X_train))
    print('logistic regression train AUC score:')
    print(roc_auc_score(y_train, preds))
    print('logistic regression validation AUC score:')
    lr_auc = roc_auc_score(y_val, res.predict(add_constant(X_val)))
    auc_scores = OrderedDict()
    auc_scores['logistic regression'] = lr_auc
    print(lr_auc)
    accs = OrderedDict()
    lr_val_preds = res.predict(add_constant(X_val)) > 0.5
    accs['logistic regression'] = np.mean(lr_val_preds == y_val)

    C = confusion_matrix(y_val, lr_val_preds)
    show_confusion_matrix(C, ['Denied', 'Approved!'])

    # print key for variables:
    for i, v in zip(range(1, train_dummies.shape[1] + 1), train_dummies.columns[1:]):
        print('x' + str(i).ljust(2), ':', v)

    # decision tree
    dt = DTC(random_state=42)
    dt.fit(X_train, y_train)
    print('decision tree regression train AUC score:')
    print(roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
    print('decision tree regression validation AUC score:')
    print(roc_auc_score(y_val, dt.predict_proba(X_val)[:, 1]))
    # overfitting bigtime, train is AUC of 1, val is 0.6
    # gridsearch for best params
    params = {
                'max_depth': [3, 6, 9, 12, 20],
                'min_samples_split': [2, 4, 6, 8, 10],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
    }
    gs = GSCV(dt, param_grid=params, n_jobs=-1)
    gs.fit(X_train, y_train)
    print('best decision tree hyperparameters:')
    print(gs.best_params_)
    dt = DTC(**gs.best_params_)
    dt.fit(X_train, y_train)
    print('decision tree train AUC score:')
    print(roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
    print('decision tree validation AUC score:')
    dt_auc = (roc_auc_score(y_val, dt.predict_proba(X_val)[:, 1]))
    auc_scores['decision tree'] = dt_auc
    print(dt_auc)
    dt_val_preds = dt.predict_proba(X_val)[:, 1] > 0.5
    accs['decision tree'] = np.mean(dt_val_preds == y_val)

    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:10]
    xnums = np.array(range(top_10.shape[0]))
    plt.bar(xnums, importances[top_10])
    plt.xticks(xnums + 0.5, map(lambda x: '\n'.join(x.split('_')), train_dummies.columns[1:][top_10]), rotation='vertical')
    plt.tight_layout()
    plt.show()

    C = confusion_matrix(y_val, dt.predict(X_val))
    show_confusion_matrix(C, ['Denied', 'Approved!'])

    # random forest
    rf = RandomForestClassifier(random_state=42)
    params = {
                'n_estimators': [50, 100, 200, 300],
                'max_features': [3, 5, 10, 'sqrt'],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 4, 6, 8],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
    }
    # this takes a while...
    gs = GSCV(rf, param_grid=params, n_jobs=-1)
    gs.fit(X_train, y_train)
    print('best random forest hyperparameters:')
    print(gs.best_params_)
    rf = RandomForestClassifier(**gs.best_params_)
    rf.fit(X_train, y_train)
    print('random forest train AUC score:')
    print(roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]))
    print('random forest validation AUC score:')
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    auc_scores['random_forest'] = rf_auc
    print(rf_auc)
    rf_val_preds = rf.predict_proba(X_val)[:, 1] > 0.5
    accs['random forest'] = np.mean(rf_val_preds == y_val)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:10]
    xnums = np.array(range(top_10.shape[0]))
    plt.bar(xnums, importances[top_10])
    plt.xticks(xnums + 0.5, map(lambda x: '\n'.join(x.split('_')), train_dummies.columns[1:][top_10]), rotation='vertical')
    plt.tight_layout()
    plt.show()

    C = confusion_matrix(y_val, rf.predict(X_val))
    show_confusion_matrix(C, ['Denied', 'Approved!'])

    # gradient booster
    gb = GBC(random_state=42)
    params = {
                'n_estimators': [200],
                'max_features': [3, 5, 10, 'sqrt'],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 4, 5],
                'subsample': [0.3, 0.4, 0.5, 0.6],
                'min_samples_split': [2, 4, 6, 8],
                'random_state': [42]
    }
    # also takes a long time...(minutes)
    gs = GSCV(gb, param_grid=params, n_jobs=-1)
    gs.fit(X_train, y_train)
    print('best gradient tree hyperparameters:')
    print(gs.best_params_)
    gb = GBC(**gs.best_params_)
    gb.fit(X_train, y_train)
    print('gradient forest train AUC score:')
    print(roc_auc_score(y_train, gb.predict_proba(X_train)[:, 1]))
    print('gradient forest validation AUC score:')
    gb_auc = roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1])
    auc_scores['gradient trees'] = gb_auc
    print(gb_auc)
    gb_val_preds = gb.predict_proba(X_val)[:, 1] > 0.5
    accs['gradient forest'] = np.mean(gb_val_preds == y_val)

    importances = gb.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:10]
    xnums = np.array(range(top_10.shape[0]))
    plt.bar(xnums, importances[top_10])
    plt.xticks(xnums + 0.5, map(lambda x: '\n'.join(x.split('_')), train_dummies.columns[1:][top_10]), rotation='vertical')
    plt.tight_layout()
    plt.show()

    C = confusion_matrix(y_val, gb.predict(X_val))
    show_confusion_matrix(C, ['Denied', 'Approved!'])

    # print summary scores
    for a in auc_scores:
        print(a.ljust(25) + str(round(auc_scores[a], 3)))

    for a in accs:
        print(a.ljust(25) + str((format(round(accs[a], 3), '.3f'))))


    # ROC curve for gradient booster
    from sklearn.metrics import roc_curve, auc
    y_score = gb.predict_proba(X_val)[:, 1]
    n_classes = y_val.shape[0]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # finally...predict the future
    print('number of predicted approved loans:', np.sum(gb.predict(test_dummies.values)))

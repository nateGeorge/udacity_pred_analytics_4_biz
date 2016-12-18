from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

current = pd.read_excel('p1-customers.xlsx')
new = pd.read_excel('p1-mailinglist.xlsx')

# the instructions were not clear here.  Do we fit a regression to profit,
# or gross revenue?  I'm guessing from my first failing review, it's supposed
# to be a fit to gross revenue??
#current['profit'] = current['Avg Sale Amount'] * 0.5 - 6.5
# it's also not clear if revenue is just avg sale amount, or avg sale amt * avg num products purchased
current['revenue'] = current['Avg Sale Amount']

cust_conv_dict = OrderedDict()
cust_conv_dict['Store Mailing List'] = 0
cust_conv_dict['Loyalty Club Only'] = 1
cust_conv_dict['Credit Card Only'] = 2
cust_conv_dict['Loyalty Club and Credit Card'] = 3
current['loyalty_level'] = current['Customer Segment'].apply(lambda x: cust_conv_dict[x])
new['loyalty_level'] = new['Customer Segment'].apply(lambda x: cust_conv_dict[x])

# looks like mostly mailing list customers
# current['loyalty_level'].hist(bins=4)
# plt.show()

pred_vars = ['loyalty_level', 'Avg Num Products Purchased', '# Years as Customer']
# looks like 'store mailing list' should be 0, and 'credit card only' should be 1,
# if it was to be a linear relationship with CC/mailing list, etc
# looks like num years customer doesn't matter

for p in pred_vars:
    current.plot(x=p, y='revenue', kind='scatter', alpha=0.5)
    plt.xlabel(p)
    plt.ylabel('revenue')
    if p == 'loyalty_level':
        plt.xticks(list(cust_conv_dict.values()), list(cust_conv_dict.keys()),
                    rotation=10)
    plt.show()

current = pd.get_dummies(data=current, columns=['Customer Segment'], prefix='', prefix_sep='')
new = pd.get_dummies(data=new, columns=['Customer Segment'], prefix='', prefix_sep='')
# need to drop 'Credit Card Only' so the coefficient will be 0 for that variable
# essentially, the coefficient for Credit Card Only will be rolled into the intercept
current.drop('Credit Card Only', inplace=True, axis=1)
new.drop('Credit Card Only', inplace=True, axis=1)

def fit_3_vars(current):
    # try a fit with # years customer in there to show the R2 is worse
    # takes the 'current' dataframe as an argument
    X = current[['loyalty_level',
                'Avg Num Products Purchased',
                '# Years as Customer']]
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    model = OLS(current['revenue'], X)
    res = model.fit()
    print(res.summary())

def fit_loyalty_level(current):
    # this fit, with loyalty level as a singe variable, actually fits the data best
    # takes the 'current' dataframe as an argument
    X = current[['loyalty_level',
                'Avg Num Products Purchased']]
    #X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = OLS(current['revenue'], X)
    res = model.fit()
    preds = res.predict(X)
    print(res.summary())

# for examining a fit
# plt.scatter(X.iloc[:, 1], current['revenue'], alpha=0.5, c='r')
# plt.scatter(X.iloc[:, 1], preds, alpha=0.5, c='b')
# plt.show()

# execute the main fit
X = current[['Loyalty Club Only',
            'Loyalty Club and Credit Card', 'Store Mailing List',
            'Avg Num Products Purchased']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = OLS(current['revenue'], X)
res = model.fit()
preds = res.predict(X)
print(res.summary())

# predict revenue from customers
new_X = new[['Loyalty Club Only',
            'Loyalty Club and Credit Card', 'Store Mailing List',
            'Avg Num Products Purchased']]
new_X = sm.add_constant(new_X)
new_preds = res.predict(new_X)

# calculate expected profit.

expected_revenue = new_preds * new['Score_Yes']
expected_profit = expected_revenue * 0.5 - 6.5
total_exp_profit = sum(expected_profit)

print('total expected profit:', total_exp_profit)

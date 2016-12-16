from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

current = pd.read_excel('p1-customers.xlsx')
new = pd.read_excel('p1-mailinglist.xlsx')

current['profit'] = current['Avg Sale Amount'] * 0.5 - 6.5

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
for p in pred_vars:
    current.plot(x=p, y='profit', kind='scatter')
    plt.xlabel(p)
    plt.ylabel('profit')
    if p == 'loyalty_level':
        plt.xticks(list(cust_conv_dict.values()), list(cust_conv_dict.keys()),
                    rotation=10)
    plt.show()

current = pd.get_dummies(data=current, columns=['Customer Segment'], prefix='', prefix_sep='')
new = pd.get_dummies(data=new, columns=['Customer Segment'], prefix='', prefix_sep='')
# looks like 'store mailing list' should be 0, and 'credit card only' should be 1.
# looks like num years customer doesn't matter

X = current[['loyalty_level',
            'Avg Num Products Purchased',
            '# Years as Customer']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = OLS(current['profit'], X)
res = model.fit()
print(res.summary())

X = current[['loyalty_level',
            'Avg Num Products Purchased']]
#X = sm.add_constant(X)  # Adds a constant term to the predictor
model = OLS(current['profit'], X)
res = model.fit()
preds = res.predict(X)
print(res.summary())

plt.scatter(X.iloc[:, 1], current['profit'], alpha=0.5, c='r')
plt.scatter(X.iloc[:, 1], preds, alpha=0.5, c='b')
plt.show()

X = current[['Credit Card Only', 'Loyalty Club Only',
            'Loyalty Club and Credit Card', 'Store Mailing List',
            'Avg Num Products Purchased']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = OLS(current['profit'], X)
res = model.fit()
preds = res.predict(X)
print(res.summary())

new_X = new[['Credit Card Only', 'Loyalty Club Only',
            'Loyalty Club and Credit Card', 'Store Mailing List',
            'Avg Num Products Purchased']]
new_X = sm.add_constant(new_X)
new_preds = res.predict(new_X)

print(sum(new_preds * new['Score_Yes']))

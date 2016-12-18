from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import re
import calendar as cd
import seaborn as sns

dem = pd.read_csv('p2-wy-demographic-data.csv')
niacs = pd.read_csv('p2-wy-453910-naics-data.csv')
# clean up for merging
niacs = niacs.rename(columns={'PHYSICAL CITY NAME':'City'})
niacs_abbr = niacs[['City', ]]
# census data from 2010 and 2000
census = pd.read_csv('p2-partially-parsed-wy-web-scrape.csv')
sales = pd.read_csv('p2-2010-pawdacity-monthly-sales.csv')

print(census.info()) # some NANs in the City|County fields...drop em like they're not
census = census.dropna()
print(census.iloc[6]) # noticed some weird stuff from the scrape...ugh
# clean up lazy web scrape
census['City'] = census['City|County'].apply(lambda x: x.split('|')[0])
census['City'] = census['City'].apply(lambda x: re.sub('\s*\?+', '', x))
census['County'] = census['City|County'].apply(lambda x: x.split('|')[1])
census['2014 Estimate'] = census['2014 Estimate'].apply(lambda x: re.sub(',', '', x.strip('<td>').strip('</td>')))
census['2010 Census'] = census['2010 Census'].apply(lambda x: re.sub(',', '', x.strip('<td>').strip('</td>')))
census['2000 Census'] = census['2010 Census'].apply(lambda x: re.sub(',', '', x.strip('<td>').strip('</td>')))
# remove weirdness (extra HTML tags)
census['2014 Estimate'] = census['2014 Estimate'].apply(lambda x: int(re.sub('<.*', '', x)))
census['2010 Census'] = census['2010 Census'].apply(lambda x: int(re.sub('<.*', '', x)))
census['2000 Census'] = census['2000 Census'].apply(lambda x: int(re.sub('<.*', '', x)))
census.drop('City|County', inplace=True, axis=1)


sales_cols = [x.strip() for x in sales.columns] # remove extra space from 'August '
rename_dict = {}
for old, new in zip(sales.columns, sales_cols):
    # special case for merging
    if old == 'CITY':
        new = 'City'
    rename_dict[old] = new


sales = sales.rename(columns=rename_dict)
# need to consolidate stores from multiple cities
sales_gr = sales.groupby('City').sum()
# calculate yearly sales
months = list(cd.month_name)[1:]
sales_gr['Total Pawdacity Sales'] = sales_gr[months].sum(axis=1)
sales_gr.reset_index(inplace=True)
abbrev_sales = sales_gr[['City', 'Total Pawdacity Sales']]

total = abbrev_sales.merge(census[['2010 Census', 'City']], on='City')

total = total.merge(dem[['City', 'Land Area', 'Population Density', 'Total Families', 'Households with Under 18']], on='City')

# Cheyenne is duplicated somehow
total.drop_duplicates(inplace=True)

# drop Gillette as per instructions
total = total[total['City'] != 'Gillette']

# custom plot of everything vs total sales
cols = list(total.columns)
f = plt.figure(figsize=(12, 12))
cnt = 1
for c in cols:
    if c in ['City', 'Total Pawdacity Sales']:
        continue
    ax = f.add_subplot(3, 2, cnt) # always forget, its rows, cols, plot#
    cnt += 1
    ax.scatter(total[c], total['Total Pawdacity Sales'])
    ax.set_xlabel(c)
    ax.set_ylabel('Total Pawdacity Sales')
    # label points with city name
    # for i, xy in enumerate(zip(total[c], total['Total Pawdacity Sales'])):                                       # <--
    #     ax.annotate(total['City'].iloc[i], xy=xy, textcoords='data', alpha=0.9)


plt.tight_layout()
plt.show()

# seaborn correlation plot
temp = total.copy()
rename = {}
rename['Total Pawdacity Sales'] = 'Pawdacity Sales (1K)'
temp['Total Pawdacity Sales'] = temp['Total Pawdacity Sales'] / 1000
rename['2010 Census'] = '2010 Census (1K)'
temp['2010 Census'] = temp['2010 Census'] / 1000
rename['Land Area'] = 'Land Area (1K)'
temp['Land Area'] = temp['Land Area'] / 1000
rename['Total Families'] = 'Families (1K)'
temp['Total Families'] = temp['Total Families'] / 1000
rename['Households with Under 18'] = 'under 18 houses (1K)'
temp['Households with Under 18'] = temp['Households with Under 18'] / 1000

temp = temp.rename(columns=rename)

sns.pairplot(temp.drop('City', axis=1))
plt.show()


def full_fit(total):
    # the colinearity of population measures means we only need one.  They all have
    # high p-values otherwise
    X = total[['2010 Census', 'Population Density', 'Total Families', 'Households with Under 18']]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = OLS(total['Total Pawdacity Sales'], X)
    res = model.fit()
    preds = res.predict(X)
    print(res.summary())

# execute the main fit
X = total[['2010 Census', 'Households with Under 18']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = OLS(total['Total Pawdacity Sales'], X)
res = model.fit()
preds = res.predict(X)
print(res.summary())

# predict sales in new cities
# get candidate cities that dont have pawdacity
new_df = dem.merge(census, on='City')
old_cits = list(total['City'].values)
old_cits.append('Gillette')
new_df = new_df[~new_df['City'].isin(old_cits)]
# get total competition sales
niacs_sum = niacs.groupby('City').sum()
niacs_sum = niacs_sum.reset_index()
new_df = new_df.merge(niacs_sum, on='City', how='left')
new_df['SALES VOLUME'] = new_df['SALES VOLUME'].fillna(0)
# filter using provided criteria
new_df = new_df[new_df['SALES VOLUME'] < 500000] # only dropped one
new_df = new_df[new_df['2014 Estimate'] > 4000] # whittled down to 6 cities



new_X = new_df[['2010 Census', 'Households with Under 18']]
new_X = sm.add_constant(new_X)
new_preds = res.predict(new_X)
new_df['predicted sales'] = new_preds
new_df.drop('County_x', inplace=True, axis=1)
new_df.drop('County_y', inplace=True, axis=1)
new_df.drop('2000 Census', inplace=True, axis=1)
new_df.sort_values(by='predicted sales', ascending=False)

def print_averages(total):
    '''
    prints averages for each column in the 'total' dataframe
    alse saves them in a csv file, averages.csv
    '''
    print('\naverages:\n')
    with open('averages.csv', 'w') as f:
        for c, s in zip(total.columns[1:], total.mean()):
            f.write(c + ',' + str(s) + '\n')
            print(c, s)

    print('\n')

descr = total.describe()
iqrs = []
cols = descr.columns
for c in cols:
    iqrs.append((descr.ix['75%', c] - descr.ix['25%', c]))


def outlier_analysis(total):
    '''
    performes outlier analysis on 'total' dataframe
    uses IQR to find outliers
    '''
    print('\n\nOutlier analysis:\n')
    for i, r in total.iterrows():
        for j, c in enumerate(cols):
            if r[c] > descr.ix['75%', c] + iqrs[j]:
                print(r['City'])
                print(c, 'above 1.5*IQR:', descr.ix['75%', c] + iqrs[j], 'value:', r[c], '\n')
            if r[c] < descr.ix['25%', c] - iqrs[j]:
                print(r['City'])
                print(c, 'below 1.5*IQR:', descr.ix['25%', c] - iqrs[j], 'value:', r[c], '\n')

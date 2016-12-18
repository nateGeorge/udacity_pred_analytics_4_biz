from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import re
import calendar as cd

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

with open('averages.csv', 'w') as f:
    for c, s in zip(total.columns[1:], total.mean()):
        f.write(c + ',' + str(s) + '\n')
        print(c, s)

descr = total.describe()
iqrs = []
cols = descr.columns
for c in cols:
    iqrs.append((descr.ix['75%', c] - descr.ix['25%', c]))


print('\n\nOutlier analysis:\n')
for i, r in total.iterrows():
    for j, c in enumerate(cols):
        if r[c] > descr.ix['75%', c] + iqrs[j]:
            print(r['City'])
            print(c, 'above 1.5*IQR:', descr.ix['75%', c] + iqrs[j], 'value:', r[c], '\n')
        if r[c] < descr.ix['25%', c] - iqrs[j]:
            print(r['City'])
            print(c, 'below 1.5*IQR:', descr.ix['25%', c] - iqrs[j], 'value:', r[c], '\n')

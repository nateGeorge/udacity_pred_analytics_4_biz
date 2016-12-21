# problems with reading csvs with pandas 0.19.0 in python 3.
# worked in python 2 with pandas 0.18.1
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.metrics import silhouette_score
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean

country_df = pd.read_csv('country-data.csv')
#print country_df.info() # my god... so many nans... the hunanity!
# blank row for some reason, drop it like its not
country_df.dropna(subset=['Country Code'], inplace=True)
# drop all 'Background' and 'Health' variables
to_drop = ['IT_NET_USER_P2',
            'SH_DYN_AIDS_ZS',
            'SH_DYN_MORT',
            'SH_MED_PHYS_ZS',
            'SH_XPD_PCAP',
            'SN_ITK_DEFC_ZS',
            'SP_POP_DPND',
            'SG_VAW_BURN_ZS',
            'SH_TBS_PREV']


for c in to_drop:
    country_df.drop(c, axis=1, inplace=True)

# drop countries with
country_df.dropna(thresh=46, inplace=True) # thresh is number of non-na, so
# drop any with more than 25 NA (71-25 = 46 non-na values)

# count number of non-null and make bar plot
plt.rc('font', size=10) # too many countries for normal font size
non_null = country_df.set_index('Country Name').count(axis=1)
non_null.sort()
non_null = pd.DataFrame(non_null)
non_null = non_null.rename(columns={0:'non-null points'})
non_null.plot.barh(figsize=(12, 24))
plt.tight_layout()
f = plt.gcf()
f.savefig('non-null_counts.png')

# try different clustering
y = country_df['Country Name'].values
cols = country_df.columns
cluster_data = country_df[cols[4:]].values
imputer = Imputer()
X = imputer.fit_transform(cluster_data)

# create clustering estimators
X = StandardScaler().fit_transform(X)
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)
kmeans = KMeans(n_clusters=4)
spectral = cluster.SpectralClustering(n_clusters=4,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
ward = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward',
                                       connectivity=connectivity)
average_linkage = cluster.AgglomerativeClustering(
    linkage="average", n_clusters=4,
    connectivity=connectivity)
complete_linkage = cluster.AgglomerativeClustering(
    linkage="complete", n_clusters=4,
    connectivity=connectivity)
birch = cluster.Birch(n_clusters=4)

clustering_names = [
    'k-means',
    'SpectralClustering', 'Ward', 'AvgLinkage', 'CompleteLinkage',
    'Birch']
clustering_algorithms = [
    kmeans, spectral, ward, average_linkage,
    complete_linkage, birch]

plot_num = 1

# pca for plotting

pca = PCA(n_components=2)
country_pca = pca.fit_transform(X)

colors = list('bgcmkbgrcmykbgrcmykbgrcmyk')

# prototype first with k-means
def test_clustering():
    km = KMeans(n_clusters=4)
    km.fit(X)
    y_pred = km.labels_.astype(np.int)
    c_list = []
    for y_ in y_pred:
        c_list.append(colors[y_])

    plt.scatter(country_pca[:, 0], country_pca[:, 1], color=c_list, s=10)
    center_colors = colors[:len(centers)]
    plt.show()

plot_num = 1
sil_scores = []
for name, algorithm in zip(clustering_names, clustering_algorithms):
    # predict cluster memberships
    t0 = time.time()
    algorithm.fit(X)
    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    sil_scores.append(silhouette_score(X, y_pred, metric='euclidean'))

    c_list = []
    for y_ in y_pred:
        c_list.append(colors[y_ - 1])

    # plot
    plt.subplot(2, 3, plot_num)
    plt.title(name, size=18)
    plt.scatter(country_pca[:, 0], country_pca[:, 1], color=c_list, s=10)

    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num += 1

plt.show()


# do PCA on Education_Avg Years, Education_Pct, and Education_Literacy groups
ed_avg_col = []
ed_pct_col = []
ed_lit_col = []
for c in country_df.columns:
    if 'BAR_SCHL' in c:
        ed_avg_col.append(c)
    elif 'UIS_EA' in c:
        ed_pct_col.append(c)
    elif 'SE_ADT' in c:
        ed_lit_col.append(c)

ed_avg_col = np.array(ed_avg_col)
ed_pct_col = np.array(ed_pct_col)
ed_lit_col = np.array(ed_lit_col)

# make PCA for each category and plot explained variance
ed_avg_pca = PCA(n_components=len(ed_avg_col))
ed_avg = ed_avg_pca.fit_transform(country_df[ed_avg_col])
ed_avg_ev = ed_avg_pca.explained_variance_ratio_
# ed_avg_idx = np.argsort(ed_avg_ev)[::-1] # don't need to do this, it's already sorted
xs = range(ed_avg_col.shape[0])
plt.bar(xs, np.cumsum(ed_avg_ev))
plt.xlabel('PCA dimension')
plt.ylabel('cumultavie explained variance ratio')
plt.ylim([0.9, 1])
plt.title('Education Average PCA')
plt.tight_layout()
plt.show()
# 3 components gets about 98% of variance

ed_pct_pca = PCA(n_components=len(ed_pct_col))
pct_imputer = Imputer() # need to impute some nans here
ed_pct = ed_pct_pca.fit_transform(pct_imputer.fit_transform(country_df[ed_pct_col]))
ed_pct_ev = ed_pct_pca.explained_variance_ratio_
#ed_pct_idx = np.argsort(ed_pct_ev)[::-1]
xs = range(ed_pct_col.shape[0])
plt.bar(xs, np.cumsum(ed_pct_ev))
plt.xlabel('PCA dimension')
plt.ylabel('cumultavie explained variance ratio')
plt.ylim([0.6, 1])
plt.title('Education Percent PCA')
plt.tight_layout()
plt.show()
# 7 components gets about 95% of variance

ed_lit_pca = PCA(n_components=len(ed_lit_col))
pct_imputer = Imputer() # again, nans
ed_lit = ed_lit_pca.fit_transform(pct_imputer.fit_transform(country_df[ed_lit_col]))
ed_lit_ev = ed_lit_pca.explained_variance_ratio_
#ed_lit_idx = np.argsort(ed_lit_ev)[::-1]
xs = range(ed_lit_col.shape[0])
plt.bar(xs, np.cumsum(ed_lit_ev))
plt.xlabel('PCA dimension')
plt.ylabel('cumultavie explained variance ratio')
plt.ylim([0.9, 1])
plt.title('Education Literacy PCA')
plt.tight_layout()
plt.show()
# looks like PCA dimensions up to 3 explains 99% of the variance.


# transform data into PCA
ed_avg_data = ed_avg[:, 0:3]
ed_pct_data = ed_pct[:, 0:7]
ed_lit_data = ed_lit[:, 0:3]

all_pca_set = set(ed_avg_col) | set(ed_pct_col) | set(ed_lit_col)
label_set = set(['Country Name', 'Country Code', 'Long Name', 'Table Name'])
full_col_set = set(country_df.columns)
cols_left = list(full_col_set.difference(all_pca_set).difference(label_set))
rest_of_data = country_df[cols_left]
X = np.hstack((rest_of_data, ed_avg_data))
X = np.hstack((X, ed_pct_data))
X = np.hstack((X, ed_lit_data))
# impute nans
final_imputer = Imputer()
X_imp = final_imputer.fit_transform(X)
X_imp = StandardScaler().fit_transform(X_imp)
col_list = cols_left + ['Ed_avg PCA' + str(x + 1) for x in range(ed_avg_data.shape[1])] \
    + ['Ed_pct PCA' + str(x + 1) for x in range(ed_pct_data.shape[1])] \
    + ['Ed_lit PCA' + str(x + 1) for x in range(ed_lit_data.shape[1])]

# use PCA + other data to do final clustering
f_pca = PCA(n_components=X_imp.shape[1])
X_pca = f_pca.fit_transform(X_imp)
plot_num = 1
sil_scores = []
for name, algorithm in zip(clustering_names, clustering_algorithms):
    # predict cluster memberships
    t0 = time.time()
    algorithm.fit(X_imp)
    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X_imp)

    sil_scores.append(silhouette_score(X_imp, y_pred, metric='euclidean'))

    c_list = []
    for y_ in y_pred:
        c_list.append(colors[y_ - 1])

    # plot
    plt.subplot(2, 3, plot_num)
    plt.title(name, size=18)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color=c_list, s=10)

    plt.xticks(())
    plt.yticks(())
    if plot_num == 4:
        print 'should be setting axis label'
        plt.xlabel('PCA dimension 1')
        plt.ylabel('PCA dimension 2')

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num += 1

plt.show()

# use silhouette_scores to compare methods
for name, sil_score in zip(clustering_names, sil_scores):
    print name, ('{0:.2f}').format(sil_score)

# get USA cluster
km = KMeans(n_clusters=4)
km.fit(X_imp)
labels = km.labels_
country_df.reset_index(drop=True, inplace=True)
USA_mask = country_df[country_df['Country Name'] == 'United States'].index[0]
USA_label = labels[USA_mask] # I got group 0 the first time
cl_mask = labels == USA_label


# get nearest countries to the USA
def calc_cos(row):
    print row
    return cosine(dist_data.ix[USA_mask][['FB_ATM_TOTL_P5',
    'IC_TAX_TOTL_CP_ZS']].values, row[['FB_ATM_TOTL_P5',
    'IC_TAX_TOTL_CP_ZS']].values)
dist_data = country_df[cl_mask][['Country Name', 'FB_ATM_TOTL_P5', 'IC_TAX_TOTL_CP_ZS']]
dist_data.dropna(inplace=True)
dist_data['cos_dist'] = 1.0
dist_data['my_cos_dist'] = 1.0
# couldn't figure this out with .apply
USA_pt = dist_data.ix[USA_mask][['FB_ATM_TOTL_P5', 'IC_TAX_TOTL_CP_ZS']].values

# cosine distance gave weird results... go with euclidean
def calc_cos(u1, u2):
    return 1- np.dot(u1, u2) / abs(np.linalg.norm(u1) * np.linalg.norm(u2))

for i, r in dist_data.iterrows():
    # cosine calculates cosine similarity, NOT cosine distance like the docs say!
    data_dist = dist_data.set_value(i, 'cos_dist', cosine(USA_pt, r[['FB_ATM_TOTL_P5', \
        'IC_TAX_TOTL_CP_ZS']].values))

    print calc_cos(USA_pt, r[['FB_ATM_TOTL_P5', \
        'IC_TAX_TOTL_CP_ZS']].values)
    data_dist = dist_data.set_value(i, 'my_cos_dist', calc_cos(USA_pt, r[['FB_ATM_TOTL_P5', \
        'IC_TAX_TOTL_CP_ZS']].values))

    data_dist = dist_data.set_value(i, 'euc_dist', euclidean(USA_pt, r[['FB_ATM_TOTL_P5', \
        'IC_TAX_TOTL_CP_ZS']].values))




# get closest 4 countries
top4 = data_dist.sort_values(by='euc_dist').iloc[1:5]['Country Name'].values # , ascending=False
for i, f in enumerate(top4):
    print str(i + 1) + '.', f

# plot tax rate by ATM machines for USA cluster
data_dist.plot.scatter('FB_ATM_TOTL_P5', 'IC_TAX_TOTL_CP_ZS')
USA = country_df.iloc[USA_mask]
plt.annotate('USA', xy = (USA['FB_ATM_TOTL_P5'], USA['IC_TAX_TOTL_CP_ZS']))#, xytext = (-20, 20))
for f in top4:
    co = dist_data[dist_data['Country Name'] == f]
    plt.annotate(f, xy = (co['FB_ATM_TOTL_P5'], co['IC_TAX_TOTL_CP_ZS']))#, xytext = (-20, 20))
plt.show()

# print sorted names from USA clusters
for n in sorted(data_dist['Country Name'].values):
    print n

# export clusters with country names for tableau map
data_dist.to_csv('USA_cluster.csv')

full_df = data_dist.merge(country_df, on='Country Name', how='left')
full_df.to_csv('USA_cluster_full.csv')

import pandas as pd
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
import numpy as np

#Requires plotly account and API key
#plotly.tools.set_credentials_file(username='username', api_key='password')


df = pd.read_csv(
    filepath_or_buffer='ecoli.data', 
    header=None, 
    sep='\t')

df.columns=['Name', 'McGeoch', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

df.dropna(how="all", inplace=True) # drops the empty line at file-end

X = df.loc[:,['McGeoch', 'gvh', 'aac', 'alm1']].values
y = df.iloc[:,8].values

traces = []

legend = {0:True, 1:False, 2:False, 3:False, 4:False, 5:False, 6:False, 7:False}

colors = {'cp': 'rgb(31, 119, 180)', 
            'im': 'rgb(255, 127, 14)', 
            'pp': 'rgb(44, 160, 44)',
            'imU': 'rgb(255, 13, 255)',
            'om': 'rgb(73, 97, 255)',
            'omL': 'rgb(255, 207, 13)',
            'imL': 'rgb(62, 255, 95)',
            'imS': 'rgb(13, 129, 255)'}


for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col], 
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)

layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.15], title='AAA'),
                xaxis2=XAxis(domain=[0.2, 0.35], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.40, 0.55], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.6, 0.75], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='E coli')

fig = Figure(data=data, layout=layout)
py.iplot(fig)

X_std = StandardScaler().fit_transform(X)

for ev in u:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,5)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,5)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.iplot(fig)

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


Y = X_std.dot(matrix_w)

traces = []

for name in ('cp', 'im','pp','imU','om','omL','imL','imS'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.iplot(fig)
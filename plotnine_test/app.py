# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from io import BytesIO
import base64
import pandas as pd
import numpy as np
from plotnine import *

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



wine=datasets.load_wine()


df=pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_target=pd.DataFrame(data=pd.Series(wine.target), columns=["target"])
arr_std=StandardScaler().fit_transform(df)
pca=PCA()
pc_values=pca.fit_transform(arr_std)
labels=["PC" + str(i+1) for i in range(pc_values.shape[1])]
df_pc=pd.DataFrame(data=pc_values,columns=labels)



def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.save(out_img, format='png', **save_args)
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    out_label="data:image/png;base64,{}".format(encoded)
    return out_label

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )



app.layout = html.Div(children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center'
        }
    ),
    dcc.Slider(
        id='my-id',
        min=3,
        max=7,
        value=4,
        marks={i: str(i) for i in range(3,8)}

    ),

    html.Div(id='my-div'),
    html.Img(id='cur_plot', src='')


    ])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return input_value


@app.callback(
    Output(component_id='cur_plot', component_property='src'),
    [Input(component_id='my-div', component_property='children')]
)

def update_plot(input_value):
    n_clus = int(input_value)
    include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    col_num = arr_std.shape[1]
    cols = [i for i in range(col_num)]
    cols_selected = [cols[i] for i in include]
    kmeans = KMeans(n_clusters=n_clus, random_state=10).fit(arr_std[:, cols_selected])
    df_kclus = pd.DataFrame(data=kmeans.labels_, columns=["clusters_kmeans"])

    df_final = pd.concat([df.reset_index(drop=True), df_pc.reset_index(drop=True),
                          df_target.reset_index(drop=True),
                          df_kclus.reset_index(drop=True)], axis=1)

    out_plot = (ggplot(data=df_final) +
                geom_point(mapping=aes(x="PC1", y="PC2", color="clusters_kmeans")))

    return fig_to_uri(out_plot)

if __name__ == '__main__':
    app.run_server(debug=True)
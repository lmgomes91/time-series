import plotly.express as px
import pandas as pd


def values_line_graph(data: pd.DataFrame) -> None:
    fig = px.line(data.loc[:, ['open', 'high', 'low']])
    fig.show()

import plotly.express as px
import pandas as pd


def values_line_graph(data: pd.DataFrame) -> None:
    fig = px.line(data.loc[:, ['close']])
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(
        width=800,
        height=400,
    )
    fig.show()

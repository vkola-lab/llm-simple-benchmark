import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style='whitegrid')

benchmark_wide = pd.read_csv('benchmark_summaries/benchmark_summaries_wide.csv')

fig = px.line_polar(
    benchmark_wide,
    r='pass@1_mean',
    theta='benchmark',
    color='model',
    line_close=True,
    template='plotly_white',
    title='pass@1'
)

fig.update_traces(fill = 'toself',opacity=0.5)

fig.update_layout(width=1000,height=1000)

fig.write_image('benchmark_summaries/pass@1_radar.png')

fig = px.line_polar(
    benchmark_wide,
    r='consensus_accuracy',
    theta='benchmark',
    color='model',
    line_close=True,
    template='plotly_white',
    title='cons@5'
)

fig.update_traces(fill='toself',opacity=0.5)

fig.update_layout(width=1000,height=1000)

fig.write_image('benchmark_summaries/consensus_accuracy_radar.png')
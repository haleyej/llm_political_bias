import random
import pandas as pd
import altair as alt
import streamlit as st


def model_type(s):  
    if s == 'roberta-base':
        return 'Base Model'   
    elif 'center' in s:
        return 'Centrist'
    elif 'right' in s:
        return 'Right Leaning'
    elif 'left' in s:
        return 'Left Leaning'
    

def generate_compass_plot(eval_df:pd.DataFrame):
    eval_df['model_type'] = eval_df['model'].apply(model_type)
    eval_df['economic_jitter'] = eval_df['economic'].apply(lambda s: s + (random.random() / 2))
    eval_df['social_jitter'] = eval_df['social'].apply(lambda s: s + (random.random() / 2))

    compass_df = pd.DataFrame({'x1': [-10], 'x2': [10], 'y1': [-10], 'y2': [10]})

    compass_base = alt.Chart(compass_df
        ).mark_rect(fill='none', stroke='black', strokeWidth=2.5
        ).encode(
            alt.X('x1', axis = None),
            alt.Y('y1', axis = None),
            x2='x2',
            y2='y2'
    )

    auth_left_df = pd.DataFrame({'x1': [-10], 'x2': [0], 'y1': [0], 'y2': [10]})
    auth_right_df = pd.DataFrame({'x1': [0], 'x2': [10], 'y1': [0], 'y2': [10]})
    lib_left_df = pd.DataFrame({'x1': [-10], 'x2': [0], 'y1': [-10], 'y2': [0]})
    lib_right_df = pd.DataFrame({'x1': [0], 'x2': [10], 'y1': [-10], 'y2': [0]})

    auth_left = alt.Chart(auth_left_df).mark_rect(fill='#FF7676', stroke='black', opacity=0.5).encode(
        alt.X('x1', axis = None),
        alt.Y('y1', axis = None),
        x2='x2',
        y2='y2'
    )

    auth_right = alt.Chart(auth_right_df).mark_rect(fill='#40ACFF', stroke='black', opacity=0.5).encode(
        alt.X('x1', axis = None),
        alt.Y('y1', axis = None),
        x2='x2',
        y2='y2'
    )

    lib_left = alt.Chart(lib_left_df).mark_rect(fill='#C19BEB', stroke='black', opacity=0.5).encode(
        alt.X('x1', axis = None),
        alt.Y('y1', axis = None),
        x2='x2',
        y2='y2'
    )

    lib_right = alt.Chart(lib_right_df).mark_rect(fill='#9BEE98', stroke='black', opacity=0.5).encode(
        alt.X('x1', axis = None),
        alt.Y('y1', axis = None),
        x2='x2',
        y2='y2'
    )

    compass = alt.layer(compass_base, auth_left, auth_right, lib_left, lib_right)

    domain = sorted(eval_df['model_type'].unique())
    range_ = ['dim_gray', 'seagreen', 'navy', 'firebrick']


    dots = alt.Chart(eval_df).mark_point(color='black', size=100, strokeWidth=5).encode(
                alt.X('economic_jitter', axis = None),
                alt.Y('social_jitter', axis = None), 
                color = alt.Color('model_type').title('Model Type').scale(domain=domain, range=range_),
                tooltip = [alt.Tooltip('model', title = 'Model'), 
                        alt.Tooltip('economic', title = 'Economic'), 
                        alt.Tooltip('social', title = 'Social')
                        ])
    
    final = alt.layer(compass, dots
            ).configure_view(
                stroke=None,
                strokeOpacity=0
            ).configure_axis(
                grid=False, 
                domain=False
            )

    return final



def main():
    eval_df = pd.read_csv('political_compass_scores.csv')

    st.title('Political Bias in Large Language Models')
    st.text('Final project for EECS 592, Foundations of Artificial Intelligence at the University of Michigan')
    st.text('Haley Johnson')

    chart = generate_compass_plot(eval_df)

    chart

    st.header('Raw Scores')
    st.table(eval_df)

if __name__ == '__main__':
    main()
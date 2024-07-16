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
    eval_df['economic_jitter'] = eval_df['economic'].apply(lambda s: s + (random.random() / 5))
    eval_df['social_jitter'] = eval_df['social'].apply(lambda s: s + (random.random() / 5))

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
    range_ = ['black', 'seagreen', 'navy', 'firebrick']


    dots = alt.Chart(eval_df).mark_point(color='black', size=100, strokeWidth=5).encode(
                alt.X('economic_jitter', axis = None),
                alt.Y('social_jitter', axis = None), 
                color = alt.Color('model_type', 
                                  legend = alt.Legend(orient = 'bottom')).title('Model Type').scale(domain=domain, range=range_),
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
            ).properties(
                width = 450, 
                height = 400
            )

    return final



def main():
    eval_df = pd.read_csv('evaluation/political_compass_scores.csv')

    st.set_page_config(
        layout='centered',
        page_icon='🎈'
    )

    # header
    st.title('Political Bias in Large Language Models')
    st.write('Final project for EECS 592, Foundations of Artificial Intelligence, at the University of Michigan')
    st.write('Haley Johnson')

    # methods 
    st.header('Methodology')

    # chart 
    st.header('Fine Tuning Can Induce Bias in Pre-Trained Language Models')
    chart = generate_compass_plot(eval_df)
    st.altair_chart(chart, use_container_width = True)
    st.write('Note that some noise has been added to data points to prevent overlap and improve chart readability')

    # raw data
    st.header('Raw Scores')
    st.table(eval_df)

if __name__ == '__main__':
    main()
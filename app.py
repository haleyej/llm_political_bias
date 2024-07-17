import os
import json
import random
import pandas as pd
import altair as alt
import streamlit as st
from transformers import pipeline, RobertaTokenizerFast, AutoModelForMaskedLM


def load_eval_statements(path:str) -> list[str]:
    statements = []
    with open(path) as f: 
        lines = json.loads(f.read())
        for line in lines:
            statements.append(line['statement'])
    return statements


def model_type(s):  
    if s == 'roberta-base':
        return 'Base Model'   
    elif 'center' in s:
        return 'Centrist'
    elif 'right' in s:
        return 'Right Leaning'
    elif 'left' in s:
        return 'Left Leaning'
    

def generate_compass_plot(eval_df: pd.DataFrame):
    eval_df['model_type'] = eval_df['model'].apply(model_type)
    eval_df['economic_jitter'] = eval_df['economic'].apply(lambda s: s + (random.random() / 3))
    eval_df['social_jitter'] = eval_df['social'].apply(lambda s: s + (random.random() / 3))

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


    dots = alt.Chart(eval_df).mark_point(color='black', size=150, strokeWidth=5).encode(
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
            ).interactive()

    return final


def set_up_model(model_path:str):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    mask_fill = pipeline('fill-mask', model = model, tokenizer = tokenizer, top_k = 1)
    return mask_fill


def load_methodology_statement(path:str) -> str:
    with open(path) as f:
        text = f.read().strip()
    return text


def main():
    # load helper files
    eval_statements = load_eval_statements('evaluation/political_compass.jsonl')
    st.write(eval_statements)
    eval_df = pd.read_csv('evaluation/political_compass_scores.csv')
    eval_df_plain = eval_df.copy(deep = True)
    eval_df_plain.columns = ['Model', 'Economic Score', 'Social Score']
    methods = load_methodology_statement('evaluation/methods.txt')

    # set up models
    news_right = set_up_model('haleyej/news-right')
    reddit_left = set_up_model('haleyej/reddit-left')
    roberta_base = set_up_model('roberta-base')

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
    st.write(methods)
    st.write('For more details see [these slides](https://github.com/haleyej/eecs_592_project/blob/main/slides.pdf)')

    # chart 
    st.header('Fine Tuning Can Induce Bias in Pre-Trained Language Models')
    chart = generate_compass_plot(eval_df)
    st.altair_chart(chart, use_container_width = True)
    st.write('Note that some noise has been added to the data to mitigate overlapping points and improve chart readability')

    # raw data
    st.subheader('Raw Scores')
    st.table(eval_df_plain)

    # model interactive
    st.header('Political Models')
    statement = st.selectbox('Select a political compass question: ', eval_statements)

    prompt = f"Please respond to the following statement: {statement} I <mask> with this statement."
    st.write(f'**Prompt**: Please respond to the following statement: {statement} I **<mask>** with this statement.')
    st.write('The model will replace **<mask>** with what is thinks the most likely missing word is')

    new_right_response = news_right(prompt)[0].get('token_str', '').strip()
    reddit_left_response = reddit_left(prompt)[0].get('token_str', '').strip()
    roberta_base_response = roberta_base(prompt)[0].get('token_str', '').strip()

    st.subheader('Language Model Predictions')

    st.write(f'RoBERTa Base Model (No Finetuning): I **{new_right_response}** with this statement')
    st.write(f'Right Leaning News: I **{reddit_left_response}** with this statement')
    st.write(f'Right Leaning News: I **{roberta_base_response}** with this statement')


if __name__ == '__main__':
    main()
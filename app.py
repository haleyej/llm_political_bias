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
                                  legend = alt.Legend(orient = 'bottom')).title('Model Type').scale(domain = domain, range = range_),
                tooltip = [alt.Tooltip('model', title = 'Model'), 
                        alt.Tooltip('economic', title = 'Economic'), 
                        alt.Tooltip('social', title = 'Social')
                        ])
    
    final = alt.layer(compass, dots
            ).configure_view(
                stroke = None,
                strokeOpacity = 0
            ).configure_axis(
                grid = False, 
                domain = False
            ).properties(
                width = 450, 
                height = 400
            ).interactive()

    return final


def generate_divergence_chart(resps_df:pd.DataFrame):
    # manipulate data
    resps_df = pd.melt(resps_df, id_vars = ['statement'], 
                    value_vars = ['reddit-left', 'reddit-right', 'reddit-center', 'roberta-base', 'news-left', 'news-right', 'news-center'])
    
    counts = resps_df.groupby(['statement', 'value']).count().unstack(fill_value=0).stack().reset_index()
    counts.columns = ['statement', 'response', 'count']

    # set up  interactive widgets
    statements = counts['statement'].unique()
    statements_dropdown = alt.binding_select(options = statements, name = 'Statement')
    statement_select = alt.selection_point(fields = ['statement'], value = statements[0], 
                                           bind = statements_dropdown, empty = False)

    # base chart 
    base = alt.Chart(counts).mark_bar().encode(
        alt.X('response', title = 'Response', axis = alt.Axis(titleFontSize = 15, labelFontSize = 13)), 
        alt.Y('count:Q', title = 'Count (Out of 7 Models)', axis = alt.Axis(titleFontSize = 15, labelFontSize = 12, format = 'd')), 
        color = alt.Color('response', legend = None).scale(scheme = 'tealblues'), 
        tooltip = [alt.Tooltip('count', title = 'Number of Models')]
    )

    # add interactivity
    final = base.add_params(
            statement_select
        ).transform_filter(
            statement_select
        ).properties(title = alt.TitleParams("Model Agreement by Question", 
                                            subtitle = 'The 7 models evaluated show divergent beliefs on most political compass test questions', 
                                            anchor = 'start', 
                                            fontSize = 25,
                                            subtitleFontSize = 18,
                                            dx = 15, 
                                            dy = -7.5),
                    height = 700, width = 700)
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
    eval_df = pd.read_csv('evaluation/political_compass_scores.csv')
    eval_df_plain = eval_df.copy(deep = True)
    eval_df_plain.columns = ['Model', 'Economic Score', 'Social Score']
    methods = load_methodology_statement('evaluation/methods.txt')

    resps_df = pd.read_csv('evaluation/political_compass_results.csv')

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
    st.write('#')
    st.header('Methodology')
    st.write(methods)
    st.write('For more details see [these slides](https://github.com/haleyej/eecs_592_project/blob/main/slides.pdf)')

    # compass chart 
    st.write('#')
    st.header('Fine Tuning Can Induce Bias in Pre-Trained Language Models')
    compass_chart = generate_compass_plot(eval_df)
    st.altair_chart(compass_chart, use_container_width = True)
    st.write('Note that some noise has been added to the data to mitigate overlapping points and improve chart readability')

    # raw data
    st.markdown('#')
    st.subheader('Raw Scores')
    st.table(eval_df_plain)
    st.write('While models exhibited divergent behavior, for the most part they were clustered around the center of the compass and did not display extreme views. We suspect that if we had greated computational resources and were able to fine-tune for more epochs, we could induce stronger political ideology.')

    # divergence 
    st.markdown('#')
    st.header('Model Divergence')
    divergence_chart = generate_divergence_chart(resps_df)
    st.altair_chart(divergence_chart)


    # model interactive
    st.markdown('#')
    st.header('The Political Compass Test as a Masked Language Modeling Task')
    st.write('The political compass test consists of 62 questions, where users are asked if they agree, strongly agree, disagree, or strongly disagree. Because answers are constrained to these four choices, it is easy to translate the political compass test into a masked language modeling task.')
    
    # ask it a question
    st.markdown('#')
    st.subheader('Language Model Predictions')
    statement = st.selectbox('Select a political compass question: ', eval_statements)

    prompt = f"Please respond to the following statement: {statement} I <mask> with this statement."
    st.write(f'**Prompt**: Please respond to the following statement: {statement} I **<mask>** with this statement.')
    st.write('The model will replace **<mask>** with what is thinks the most likely missing word is')

    new_right_response = news_right(prompt)[0].get('token_str', '').strip()
    reddit_left_response = reddit_left(prompt)[0].get('token_str', '').strip()
    roberta_base_response = roberta_base(prompt)[0].get('token_str', '').strip()
    st.write(f'RoBERTa Base Model (No Finetuning): I **{new_right_response}** with this statement')
    st.write(f'Left Leaning Reddit Posts: I **{reddit_left_response}** with this statement')
    st.write(f'Right Leaning News: I **{roberta_base_response}** with this statement')

    # scoring explanation
    st.markdown('#')
    st.subheader('Scoring')
    st.write('''Responses are categorized into agree, strongly agree, disagree, and stronly disagree based on \n 1) the most likely token, 
             \n 2) the probability of the top token,
             \n 3) the probability of other tokens. 
             \n For instance, if in a given model there was a high probability of filling the **<MASK>** token with 
             **agree** and a very low probability of filling it with **disagree**, that response may be classified as "strongly agree," even though that 
             was not the exact token the model predicted.''')
    
    # ask your own question 
    st.markdown('#')
    st.subheader('Ask Your Own Question')
    q = st.text_input('Input here: ', value = 'immigration is positive for society')

    user_prompt = f"Please respond to the following statement: {q} I <mask> with this statement."
    st.write(f'**Prompt**: Please respond to the following statement: {q}. I **<mask>** with this statement.')

    #user_prompt

    new_right_response_q = news_right(user_prompt)[0].get('token_str', '').strip()
    reddit_left_response_q = reddit_left(user_prompt)[0].get('token_str', '').strip()
    roberta_base_response_q = roberta_base(user_prompt)[0].get('token_str', '').strip()
    st.markdown('#### Results')
    st.write(f'RoBERTa Base Model (No Finetuning): I **{new_right_response_q}** with this statement')
    st.write(f'Left Leaning Reddit Posts: I **{reddit_left_response_q}** with this statement')
    st.write(f'Right Leaning News: I **{roberta_base_response_q}** with this statement')



if __name__ == '__main__':
    main()
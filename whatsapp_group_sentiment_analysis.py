import pandas as pd
import numpy as np
from collections import defaultdict
import streamlit as st
from transformers import AutoTokenizer, AutoModel, pipeline


# Read uploaded file from user
def get_total_chat():
  with open('chat.txt') as file:
    lines = file.readlines()
  if not lines:
    return
  if len(lines) < 2:
    return
  file = st.file_uploader('chat.txt')
  if not file:
    return None
  lines = file.readlines()
  return [line.decode("utf-8") for line in lines]
  

# Create a people dictionary of lists where every person's name is the key and the list of his msgs is the value
def get_people(total_chat):
  people = defaultdict(list)
  for l in total_chat:
    sen = l.rstrip("\n")
    iphone_whatsapp_separator = ']'
    android_whatsapp_separator = ' - '
    if iphone_whatsapp_separator in sen:
      name_and_msg = sen.split(iphone_whatsapp_separator)[1]
    elif android_whatsapp_separator in sen:
      name_and_msg = sen.split(android_whatsapp_separator)[1]
    if ':' in name_and_msg:
        name_and_msg = name_and_msg.strip()
        name = name_and_msg.split(':')[0].replace(' ', '_')
        msg = name_and_msg.split(':')[1]
        people[name].append(msg)
  # Clean noise made by the subject of the group
  for person, msgs in people.copy().items():
    if len(msgs) < 5:
      people.pop(person)

  # Get the no. of msgs of the person with the fewest no. of msgs
  min_msgs = np.min([len(msgs) for person, msgs in people.items()])

  # Align everyone's no. of msgs to his no. of msgs so we can put in a dataframe
  for person, msgs in people.items():
    while len(msgs) > min_msgs:
      msgs.pop()

  return people


def get_positivity(string):
  states = sentiment_analysis(string)[0]
  for state in states:
    if state['label'] == 'positive':
      break
  return state['score']

def get_negativity(string):
  states = sentiment_analysis(string)[0]
  for state in states:
    if state['label'] == 'negative':
      break
  return state['score']

def get_neutrality(string):
  states = sentiment_analysis(string)[0]
  for state in states:
    if state['label'] == 'natural':
      break
  return state['score']


tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis", force_download=True)
model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis", force_download=True)

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)

def run():
  
  st.title('WhatsApp Groups Sentiments Analyzer!')
  
  total_chat = get_total_chat()
  if not total_chat:
    return
  people = get_people(total_chat)
  df = pd.DataFrame(data=people)
  
  for person in df.columns:
    df[f'{person}_positivity_index'] = df[person].apply(lambda r: get_positivity(r))
    df[f'{person}_negativity_index'] = df[person].apply(lambda r: get_negativity(r))
    df[f'{person}_neutrality_index'] = df[person].apply(lambda r: get_neutrality(r))
    
  for person in people.keys():
    df[f'{person}_positivity_score'] = ((df[f'{person}_positivity_index'] + 
                                           df[f'{person}_negativity_index']) / 2 + 
                                          df[f'{person}_neutrality_index']) / 2

  people_scores = []
  for name in people.keys():
    score = round(df[f'{name}_positivity_score'].mean(), 5)
    people_scores.append((name, score))
  people_scores.sort(reverse=True, key=lambda x:x[1])
  for person in people_scores:
    st.write((f'{person[0]} positivity score: {person[1]}'))
  

run()

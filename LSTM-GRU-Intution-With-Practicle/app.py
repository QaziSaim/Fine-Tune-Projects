import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas 

# Load LSTM MOdel
model = load_model('model/my_lstm_model.keras')

with open('model/tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)


def predict(model, tokenizer, text, max_sequenc_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequenc_len:
    token_list = token_list[-(max_sequenc_len-1):]
  token_list = pad_sequences([token_list], maxlen=max_sequenc_len-1)
  predicted = model.predict(token_list,verbose=0)
  predicted_word_index = np.argmax(predicted,axis=1)[0]
  for word,index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None


# streamlit
st.title("Next word prediction with LSTM")
input_text = st.text_input("Enter the sequence of words","To be or not to be")
if st.button("predict next word"):
    # input_text = "Mar. Horito"
    # print(f"Input text:{input_text}")
    max_sequenc_len = model.input_shape[1] + 1
    next_word = predict(model, tokenizer, input_text, max_sequenc_len)
    st.write(f'Next word: {next_word}')
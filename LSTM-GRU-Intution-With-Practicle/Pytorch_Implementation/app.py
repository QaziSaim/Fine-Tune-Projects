import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------
# Define the PyTorch Model
# -------------------
class LSTMTextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=150, hidden_dim2=100):
        super(LSTMTextGen, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]   # take last hidden state
        out = self.fc(x)
        return out

# -------------------
# Load model and tokenizer
# -------------------
with open("model/hamlet_pytorch_tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

total_words = len(tokenizer.word_index) + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMTextGen(total_words)
model.load_state_dict(torch.load("model/pytorch_lstm_model.pth", map_location=device))
model.to(device)
model.eval()

# -------------------
# Prediction function
# -------------------
def predict(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1)

    token_tensor = torch.tensor(token_list, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(token_tensor)
        predicted_word_idx = torch.argmax(output, dim=1).item()

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_idx:
            return word
    return None

# -------------------
# Streamlit App
# -------------------
st.title("Next Word Prediction with PyTorch LSTM")
input_text = st.text_input("Enter the sequence of words", "To be or not to be")

if st.button("Predict Next Word"):
    max_seq_len = model.embedding.num_embeddings  # or store from training
    max_seq_len = model.fc.out_features  # better: pass max_seq_len you used in training
    next_word = predict(model, tokenizer, input_text, max_seq_len)
    st.write(f"Next word: {next_word}")

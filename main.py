import streamlit as st
import numpy as np
import tensorflow as tf
import tiktoken
from transformer import PaddingMask,PositionalEmbedding,TransformerDecoder,TransformerEncoder

# Assume these are already loaded somewhere
@st.cache_resource
def load_chatbot_model():
    # 👇 change path to your trained model
    return tf.keras.models.load_model(
        "chatbot_5_epochs.keras",
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding,
            "PaddingMask": PaddingMask,
            "TransformerEncoder":TransformerEncoder
        }
    )


model = load_chatbot_model()

max_enc_len=30
max_dec_len=30

# ============================================
# Load tokenizer (tiktoken)
# ============================================

base_enc = tiktoken.get_encoding("cl100k_base")

special_tokens = {
    "<|pad|>": base_enc.n_vocab,
    "<|startoftext|>": base_enc.n_vocab + 1,
    "<|endoftext|>": base_enc.n_vocab + 2,
}

enc = tiktoken.Encoding(
    name="cl100k_base_custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=special_tokens,
)


# Special tokens
BOS_ID = enc.encode("<|startoftext|>", allowed_special={"<|startoftext|>"})[0]
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
PAD_ID = enc.encode("<|pad|>", allowed_special={"<|pad|>"})[0]

# Beam search decoding


def beam_search_decode(input_ids, max_len=max_dec_len,beam_width=3,):
    # Each beam: (sequence, score)
    beams = [([BOS_ID], 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score in beams:
            # If already ended, keep as is
            if seq[-1] == EOS_ID:
                all_candidates.append((seq, score))
                continue
            
            decoder_input = np.array([seq])
            encoder_input = np.array([input_ids])
            
            preds = model.predict(
                {"encoder_inputs": encoder_input, "decoder_inputs": decoder_input},
                verbose=0
            )
            
            probs = preds[0, -1, :]  # last token probabilities
            
            # Get top-k tokens
            top_k_ids = np.argsort(probs)[-beam_width:]
            
            for token_id in top_k_ids:
                new_seq = seq + [token_id]
                # Use log probability for stability
                new_score = score + np.log(probs[token_id] + 1e-9)
                all_candidates.append((new_seq, new_score))
        
        # Sort by score (higher is better)
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Optional: stop early if all beams ended
        if all(seq[-1] == EOS_ID for seq, _ in beams):
            break
    
    # Return best sequence (remove BOS and everything after EOS)
    best_seq = beams[0][0]
    
    # Remove BOS
    best_seq = best_seq[1:]
    
    # Cut at EOS if exists
    if EOS_ID in best_seq:
        best_seq = best_seq[:best_seq.index(EOS_ID)]
    
    return best_seq

# Padding
def pad_seq(ids, max_len):
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


# Streamlit UI
st.set_page_config(page_title="Transformer Chatbot", page_icon="🤖")

st.title("🤖 Transformer Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Preprocess input
    input_ids = enc.encode(user_input)
    input_ids = pad_seq(input_ids, max_enc_len)

    # Generate response
    pred_ids = beam_search_decode(input_ids, max_dec_len)
    response = enc.decode(pred_ids)

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
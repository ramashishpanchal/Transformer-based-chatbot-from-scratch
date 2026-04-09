import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.saving import register_keras_serializable


@register_keras_serializable() 
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len,**k):
        super().__init__(**k)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



@register_keras_serializable() 
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1,**k):
        super().__init__(**k)
        self.att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, padding_mask):
        attn_output = self.att(x, x,attention_mask=padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)


@register_keras_serializable() 
class TransformerDecoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1,**k):
        super().__init__(**k)
        self.self_att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)
        self.enc_dec_att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output,dec_padding_mask):
        # Masked self-attention
        att1 = self.self_att(x, x, x, use_causal_mask=True,attention_mask=dec_padding_mask)
        att1 = self.dropout1(att1)
        out1 = self.norm1(x + att1)

        # Encoder–Decoder attention
        att2 = self.enc_dec_att(query=out1,value=enc_output,key=enc_output)
        att2 = self.dropout2(att2)
        out2 = self.norm2(out1 + att2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.norm3(out2 + ffn_output)


@register_keras_serializable() 
class PaddingMask(layers.Layer):
    def call(self, x):
        # x: (batch, seq_len)
        mask = tf.not_equal(x,100277 )          # (batch, seq_len)
        mask = mask[:, tf.newaxis, :]              # (batch, 1, seq_len)
        return mask  
import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PhonemeDecoder(nn.Module):
    def __init__(self, phoneme_vocab_size=75, d_model=768, num_decoder_layers=2, nhead=12, dim_feedforward=768, dropout=0.1):
        super(PhonemeDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(phoneme_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, phoneme_vocab_size)
        self._initialize_weights()
        
    # param tgt_input repretation predicted sequence
    # param memory repretation encoder output
    def forward(self, tgt_input, memory, tgt_mask=None, memory_key_padding_mask=None):
        
        tgt_input = tgt_input.transpose(0, 1)
        memory = memory.transpose(0, 1)
        T_plabel = tgt_input.shape[0]
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_plabel)
        tgt_emb = self.embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
    
        decoder_output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, 
                                                  memory_key_padding_mask=memory_key_padding_mask)
        output = self.fc_out(decoder_output)
        output = output.transpose(0, 1)
                
        # the shape of output is (B, T, V)
        return output, decoder_output.transpose(0, 1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
            elif isinstance(m, nn.TransformerDecoderLayer):
                nn.init.xavier_uniform_(m.self_attn.in_proj_weight)
                nn.init.constant_(m.self_attn.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.self_attn.out_proj.weight)
                nn.init.constant_(m.self_attn.out_proj.bias, 0)
                nn.init.xavier_uniform_(m.linear1.weight)
                nn.init.constant_(m.linear1.bias, 0)
                nn.init.xavier_uniform_(m.linear2.weight)
                nn.init.constant_(m.linear2.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
def predict(model, memory, start_token=2, end_token=3, max_length=300, vocab_size=75):
    tgt_input = torch.tensor([[start_token]], dtype=torch.long).to(memory.device)
    output_sequence = []

    for _ in range(max_length):
        output, decoder_output = model(tgt_input, memory, tgt_mask=None)
        
        # use the last token to predict the next token
        next_token_logits = output[:, -1, :]  # (B, vocab_size)
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        output_sequence.append(next_token)
        
        # update tgt_input
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]], dtype=torch.long).to(memory.device)], dim=1)
        if next_token == end_token:
            break

    return output_sequence

def predict_batch(model, memory, batch_size, start_token=2, end_token=3, max_length=300, vocab_size=75, memory_mask=None):
    device = memory.device
    tgt_input = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)
    output_sequences = [[start_token] for _ in range(batch_size)]
    sequence_lengths = [1] * batch_size
    finished = [False] * batch_size

    for _ in range(max_length):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(memory.device)
        output, decoder_output = model(tgt_input, memory, memory_key_padding_mask=memory_mask, tgt_mask=tgt_mask)
        next_token_logits = output[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        for i, next_token in enumerate(next_tokens):
            if not finished[i]:
                output_sequences[i].append(next_token.item())
                sequence_lengths[i] += 1
                if next_token == end_token:
                    finished[i] = True
        
        if all(finished):
            break

        next_token_tensor = next_tokens.unsqueeze(1)
        tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

    max_seq_length = max(sequence_lengths)
    padded_output_sequences = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)
    for i, seq in enumerate(output_sequences):
        padded_output_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long).to(device)
    return padded_output_sequences, sequence_lengths_tensor

# # example inference
# vocab_size = 75
# d_model = 768
# seq_len = 361
# batch_size = 8
# memory = torch.rand(batch_size, seq_len, d_model)
# model = PhonemeDecoder()  # 编码器-解码器模型
# batch_size = 8  # 例如，我们希望一次处理4个序列
# result, lengths = predict_batch(model, memory, batch_size)
# print("Padded predicted sequences:", result)
# print("Sequence lengths:", lengths)
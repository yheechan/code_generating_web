import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class Seq2SeqPointCE(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.decoder = Decoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        

        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, src_len)
        :param trg: (bsz, max_len)
        :return outputs_pre, outputs_post: (bsz, src_len, 2)
        """

        bsz, max_len, = trg.shape
        assert pre_seq.shape == post_seq.shape
        bsz, src_len = pre_seq.shape 

        outputs_pre = torch.zeros(bsz, max_len, src_len, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, src_len, 1).cuda()

        # enc_out: (bsz, enc_len, enc_hid_dim)
        # prev_hidden: (n_layer, bsz, dec_hid_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        

        # first input. 
        # input = trg[0, :]
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, 2*src_len, 2)
            output, prev_hidden = self.decoder(input, prev_hidden, enc_out)
            
            # outputs_pre, outputs_post: (bsz, max_len, src_len, 2)
            outputs_pre[:, t, :, :] = output[:, :src_len, :]
            outputs_post[:, t, :, :] = output[:, src_len:, :]

            input = trg[:, t]
            
        return outputs_pre.squeeze(-1), outputs_post.squeeze(-1)


class Seq2SeqPointAttnCE(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.decoder = DecoderAttn(self.embedding, vocab_size, embedding_dim, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, src_len)
        :param trg: (bsz, max_len)
        :return outputs_pre, outputs_post: (bsz, src_len, 2)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)


        bsz, max_len, = trg.shape
        assert pre_seq.shape == post_seq.shape
        bsz, src_len = pre_seq.shape 

        outputs_pre = torch.zeros(bsz, max_len, src_len, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, src_len, 1).cuda()

        
        # enc_out: (bsz, enc_len, enc_hid_dim)
        # prev_hidden: (n_layer, bsz, dec_hid_dim)  
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        
        # first input. 
        # input = trg[0, :]
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, 2*src_len, 2)
            output, prev_hidden = self.decoder(input, prev_hidden, enc_out)

            # outputs_pre, outputs_post: (bsz, max_len, src_len, 2)
            outputs_pre[:, t, :, :] = output[:, :src_len, :]
            outputs_post[:, t, :, :] = output[:, src_len:, :]
            
            input = trg[:, t]

        return outputs_pre.squeeze(-1), outputs_post.squeeze(-1)

class Encoder(nn.Module):
    def __init__(self, embedding, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=1, drop_p=0.5, batch_first=True):
        super().__init__()
        self.embedding = embedding
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        
        
        self.pre_enc_layer = nn.GRU(input_size=emb_dim, hidden_size=self.enc_hid_dim, num_layers=num_layers, dropout=drop_p, batch_first=batch_first)
        self.post_enc_layer = nn.GRU(input_size=emb_dim, hidden_size=self.enc_hid_dim, num_layers=num_layers, dropout=drop_p, batch_first=batch_first)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, pre_seq, post_seq):
        """
        :param prefix: (bsz, len_pre)
        :param postfix: (bsz, len_post)
        :return enc_out: (bsz, enc_len, enc_hid_dim)
        :return enc_hidden: (bsz, dec_hid_dim)
        """
        assert pre_seq.size(1) == post_seq.size(1)

        pre_seq = self.dropout(self.embedding(pre_seq))
        post_seq = self.dropout(self.embedding(post_seq)) 
        
        pre_out, pre_hidden = self.pre_enc_layer(pre_seq)
        post_out, post_hidden = self.post_enc_layer(post_seq)
        # print(pre_out.shape, pre_hidden.shape)
        
        # bsz, (len_pre+len_post), enc_hid_dim
        enc_out = torch.cat((pre_out, post_out), dim = 1) 
        
        # return only last layer
        enc_hidden = torch.tanh(self.fc(torch.cat((pre_hidden[-1, :, :], post_hidden[-1, :, :]), dim = -1)))
        
        return enc_out, enc_hidden


class Decoder(nn.Module):
    def __init__(self, embedding, embedding_dim, enc_hid_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        self.embedding = embedding

        self.rnn = nn.GRU(embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        
        self.fc_hid = nn.Linear(2*dec_hid_dim, 1)
        self.fc_out = nn.Linear(2*dec_hid_dim, 1)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, enc_out):
        """
        :param input: (bsz)
        :param prev_hidden: (bsz, dec_hid_dim)
        :param enc_out: (bsz, enc_len, enc_hid_dim)
        :return output: (bsz, enc_len, 2)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (bsz, 1) for rnn
        embed = self.dropout(self.embedding(input))

        # bsz, 1, dec_hid_dim
        output, hidden = self.rnn(embed, prev_hidden.unsqueeze(0))


        # bsz, enc_len, dec_hid_dim
        output = output.repeat(1, enc_out.size(1), 1)
        
        # bsz, enc_len, dec_hid_dim + enc_hid_dim
        output = F.relu(torch.cat([enc_out, output], dim=-1))


        # bsz, enc_len, 1
        output = self.fc_hid(output) 
        output = self.fc_out(output)

        return output, hidden.squeeze(0)
        
class DecoderAttn(nn.Module):
    def __init__(self, embedding, num_embedding, embedding_dim, enc_hid_dim, dec_hid_dim, attention, num_layer=1, drop_p=0.5):
        super().__init__()
        
        self.embedding = embedding
        self.attention = attention

        self.rnn = nn.GRU(enc_hid_dim +embedding_dim, dec_hid_dim, batch_first=True, dropout=drop_p)

        self.fc_hid = nn.Linear(2*dec_hid_dim+embedding_dim+enc_hid_dim, 2*dec_hid_dim+embedding_dim+enc_hid_dim)
        self.fc_out = nn.Linear(2*dec_hid_dim+embedding_dim+enc_hid_dim, 1)
        self.dropout = nn.Dropout(drop_p)


    def forward(self, input, prev_hidden, enc_out):
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (bsz, 1) for rnn
        embed = self.dropout(self.embedding(input))

        # bsz, enc_len
        a = self.attention(prev_hidden, enc_out)
        
        # bsz, 1, enc_len
        a = a.unsqueeze(1)
        
        # bsz, 1, enc hid dim
        weighted = torch.bmm(a, enc_out)
        

        # bsz, 1, emb_dim+enc_hid_dim
        rnn_input = torch.cat((embed, weighted), dim = -1)

        # output: (bsz, 1, dec_hid_dim)
        output, hidden = self.rnn(rnn_input, prev_hidden.unsqueeze(0))
        

        # bsz, enc_len, dec_hid_dim
        output = output.repeat(1, enc_out.size(1), 1)
        weighted = weighted.repeat(1, enc_out.size(1), 1)
        embed = embed.repeat(1, enc_out.size(1), 1)


        # bsz, enc_len, 1
        # bsz, enc_len, 1
        out_fc_hid = self.fc_hid(torch.cat([enc_out, weighted, output, embed], dim=-1))
        output = self.fc_out(out_fc_hid)
        
        return output, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: (bsz, dec_hid_dim)
        :param encoder_outputs: (bsz, enc_len, enc_hid_dim)
        :return attention: (bsz, enc_len) 
        """
        
        batch_size = encoder_outputs.size(0)
        enc_len = encoder_outputs.size(1)
        
        # bsz, enc_len, dec_hid_dim
        hidden = hidden.unsqueeze(1).repeat(1, enc_len, 1)
        
        # bsz, enc_len, dec_hid_dim
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = -1))) 
        
        
        # bsz, enc_len
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

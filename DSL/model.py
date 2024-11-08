import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from super_duper_utils import p
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, bidirectional=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, bidirectional=bidirectional)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, bidirectional = bidirectional)
        self.decoder = nn.Linear(nhid, ntoken)
        self.bidirectional = bidirectional

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # input: seq_len*batch_size (token indices)
        # sopra capra ...
        # la    canta
        # panca sopra
        # la    la
        # capra panca
        # canta la 
        # sopra capra
        # ...   ...
        #
        # encoder(input) idem ma ogni elemento è sostituito da un vettore dell'embedding space
        emb = self.drop(self.encoder(input))
        # Seq.Bat.Emb>rnn>Seq.Bat.Hid (=output)
        # di default emb = hid = 200
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        
        # Here we create a view, which allows us to concatenate bidirectional layers in general manner
        #S.B.(2*)H>S.B.(2)1.H
        output = output.view(
            output.shape[0],
            output.shape[1],
            2 if self.rnn.bidirectional else 1,
            self.nhid,
        )
        # Here outputs of bidirectional RNNs are summed, you may concatenate it
        # It makes up for an easier implementation, and is another often used approach
        #S.B.(2)1.H>S.B.H
        output = output.sum(dim=2)

        # Seq.Bat.Hid>decoder>Seq.Bat.Ntokens (=decoded)
        decoded = self.decoder(output)
        # Seq.Bat.Ntokens>Seq*Bat.Ntokens
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        D = 1 if not self.bidirectional else 2
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(D*self.nlayers, bsz, self.nhid),
                    weight.new_zeros(D*self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(D*self.nlayers, bsz, self.nhid)

class RNNEncoder(nn.Module):
    """Accepts B.S by default. Returns B.H always."""

    def __init__(self, base_model_name, device, receives_batch_first=True):
        super().__init__()
        with open(base_model_name, 'rb') as f:
            lstm_grammar = torch.load(f, map_location=device)
        self.base_model = lstm_grammar
        
        assert self.base_model.rnn.hidden_size == self.base_model.nhid
        self.embedding_dimension = self.base_model.nhid
        self.receives_batch_first = receives_batch_first

    def forward(self, input, hidden):
        # input: seq_len*batch_size (token indices)
        # sopra capra ...
        # la    canta
        # panca sopra
        # la    la
        # capra panca
        # canta la 
        # sopra capra
        # ...   ...
        #
        # encoder(input) idem ma ogni elemento è sostituito da un vettore dell'embedding space
        if self.receives_batch_first:
            #input: B.S>S.B
            input = input.t()
        # S.B.Emb
        emb = self.base_model.drop(self.base_model.encoder(input))
        # Seq.Bat.Emb>rnn>Seq.Bat.D*Hid (=output)
        # di default emb = hid = 200
        output, hidden = self.base_model.rnn(emb, hidden)
        output = self.base_model.drop(output)

        #hidden: D*(num_lay default 2).B.Hid

        #S.B.(2*)H>S.B.(2)1.H
        output = output.view(
            output.shape[0],
            output.shape[1],
            2 if self.base_model.rnn.bidirectional else 1,
            self.embedding_dimension,
        )
        # Here outputs of bidirectional RNNs are summed, you may concatenate it
        # It makes up for an easier implementation, and is another often used approach
        #S.B.(2)1.H>S.B.H
        output = output.sum(dim=2)
        output = output.transpose(0,1) #B.S.H
        #D*num_lay.B.H>B.H
        #hidden = hidden.sum(dim=0)
        # return output
        return output, hidden #TODO: return hidden for encoder?
        # # Seq.Bat.Hid>decoder>Seq.Bat.Ntokens (=decoded)
        # decoded = self.decoder(output)
        # # Seq.Bat.Ntokens>Seq*Bat.Ntokens
        # decoded = decoded.view(-1, self.ntoken)
        # return F.log_softmax(decoded, dim=1), hidden

class VisionEncoder(nn.Module):
    """Needs B.C.H.W. Returns B.E always."""

    def __init__(self, base_model_name = "openai/clip-vit-base-patch32", device= 'cpu', receives_batch_first=True):
        super().__init__()
        self.base_model = CLIPVisionModel.from_pretrained(base_model_name)#, device_map=device)
        # self.processor = AutoProcessor.from_pretrained(base_model_name)#, device_map=device)
        
        self.embedding_dimension = list(self.base_model.parameters())[-1].shape[0]
        self.receives_batch_first = receives_batch_first

    def forward(self, images):
        #input: B.dim_one_image
        #inputs = self.processor(images, return_tensors="pt")
        batch_size = len(images)
        # '''pixel_values (torch.FloatTensor of shape (batch_size, num_channels, height, width)) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using AutoImageProcessor. See CLIPImageProcessor.call() for details.'''
        # needs B.C.H.W
        outputs = self.base_model(pixel_values = images)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled CLS states
        assert(len(pooled_output)==batch_size)
        
        return pooled_output


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

class ARCVisionGrammarAlignment(nn.Module): #100% does not work
    """Requires batch first!"""
    def __init__(self, codeEncoder, visionEncoder, common_dimension, 
                 #temperature=0., 
                 codeEncoderUnfrozen=True, visionEncoderUnfrozen=True):
        super().__init__()
        self.codeEncoder = codeEncoder
        self.codeEncoderUnfrozen = codeEncoderUnfrozen
        for param in self.codeEncoder.parameters():
            param.requires_grad = codeEncoderUnfrozen

        self.visionEncoder = visionEncoder
        self.visionEncoderUnfrozen = visionEncoderUnfrozen
        for param in self.visionEncoder.parameters():
            param.requires_grad = visionEncoderUnfrozen

        #self.temperature = torch.nn.Parameter(torch.tensor(float(temperature)), requires_grad = False)
        self.common_dimension = common_dimension
        self.proj_vision2common = torch.nn.Linear(self.visionEncoder.embedding_dimension, self.common_dimension)
        self.proj_code2common = torch.nn.Linear(self.codeEncoder.embedding_dimension, self.common_dimension)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.proj_vision2common.weight, -initrange, initrange)
        nn.init.zeros_(self.proj_vision2common.bias)
        nn.init.uniform_(self.proj_code2common.weight, -initrange, initrange)
        nn.init.zeros_(self.proj_code2common.bias)

    def forward(self, x, hidden):
        I_emb = self.visionEncoder(x["image"]) #B.dim_visio
        output, hidden = self.codeEncoder(x["input_ids"], hidden) #B.S>B.S.dim_code, D*num_lay.B.dim_code
        C_emb = output.sum(dim=1) #B.dim_code #TODO: makes sense????

        I_emb_com = torch.nn.functional.normalize(self.proj_vision2common(I_emb))
        C_emb_com = torch.nn.functional.normalize(self.proj_code2common(C_emb))
        
        assert I_emb_com.shape == C_emb_com.shape #B.dim_common
        
        #B.dim_com@dim_com.B = B.B
        logits = I_emb_com@C_emb_com.T #* torch.exp(self.temperature)
        labels = torch.arange(I_emb.size(0)) #B
        loss_I = F.cross_entropy(logits.T, labels)
        loss_C = F.cross_entropy(logits, labels)

        loss = (loss_I + loss_C)/2.0 

        return loss, hidden#, logits
    
class ARCVisionGrammarPrediction(nn.Module):
    """Requires batch first! Returns B*S.Tokens, Tokens are already log-softamaxed"""
    def __init__(self, codeEncoder, visionEncoder, n_tokens, 
                 #temperature=0., 
                 codeEncoderUnfrozen=True, visionEncoderUnfrozen=True):
        super().__init__()
        self.codeEncoder = codeEncoder
        self.codeEncoderUnfrozen = codeEncoderUnfrozen
        for param in self.codeEncoder.base_model.parameters():
            param.requires_grad = codeEncoderUnfrozen

        self.visionEncoder = visionEncoder
        self.visionEncoderUnfrozen = visionEncoderUnfrozen
        for param in self.visionEncoder.base_model.parameters():
            param.requires_grad = visionEncoderUnfrozen

        #self.temperature = torch.nn.Parameter(torch.tensor(float(temperature)), requires_grad = False)
        self.prediction_dimension = n_tokens
        self.proj_concatenation2predictions = torch.nn.Linear(
            self.visionEncoder.embedding_dimension + self.codeEncoder.embedding_dimension, self.prediction_dimension)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.proj_concatenation2predictions.weight, -initrange, initrange)
        nn.init.zeros_(self.proj_concatenation2predictions.bias)

    def forward(self, x, hidden):
        I_emb = self.visionEncoder(x["image"]) #B.dim_visio
        output, hidden = self.codeEncoder(x["input_ids"], hidden) #B.S>B.S.dim_code, D*num_lay.B.dim_code
        C_emb = output #B.S.dim_code        
        B,S,Ev = C_emb.shape[0],C_emb.shape[1],self.visionEncoder.embedding_dimension
        #B.S.(Ec=200) | B.(Ev=768) --concat--> B.S.(Ec+Ev) --proj--> B.S.Tokens 
        combination = torch.cat((C_emb, I_emb.unsqueeze(dim=1).expand(B,S,Ev)), dim = 2) # B.S.(Ec+Ev)
        predictions = self.proj_concatenation2predictions(combination) # B.S.Tokens
        predictions_flattened = predictions.view(-1, self.prediction_dimension) #B*S.Tokens
        return F.log_softmax(predictions_flattened, dim=1), hidden



if __name__ == "__main__":
    pass
    # gp=ARCVisionGrammarPrediction(visionEncoder=VisionEncoder(), codeEncoder=RNNEncoder(), n_tokens=129)
    # ve = VisionEncoder()
    # p<ve.embedding_dimension
    # t=torch.rand((1,3,224,224))
    # print(ve(t).shape)
    # t=torch.rand((2,3,224,224))
    # print(ve(t).shape)
    # print(ve(t)[:30])
#     grid = [[(i+j)%11 for j in range(224)] for i in range(224)]
    
#     i=to_pil_image(np.asarray(grid), cmap = ARCcmap)
# #     x=pil_to_tensor(i)
# # for a in tqdm(range(100)):
# #     x=manual_conversion(grid)
# s()
# #ARCimg = ARCimg.resize((224,224)) # returns stretched to fill 224.224 square
# ARCimg = padding(ARCimg, (224,224), fill='white') # returns efficient padded to fill 224.224
# ARCimg.save('temp_resized.png')
#     ve(t)
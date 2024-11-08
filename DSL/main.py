# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import tqdm


import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/smalltext_masked',#wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--masked', action='store_true', 
                    help='masked training objective. Requires dataset composed of (masked_sentence,label) couples')
parser.add_argument('--no-masked', dest='masked', action='store_false')
parser.set_defaults(masked=True)
parser.add_argument('--not_bidirectional', action='store_true', 
                    help='used for masked training objective. Entails RNNs are not bidirectional (by default they are)')
parser.set_defaults(not_bidirectional=False)
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


###############################################################################
# Load data
###############################################################################
def stop():
    raise TypeError('stop')

corpus = data.Corpus(args.data, masked = args.masked)

try:
    mask_index = corpus.dictionary.word2idx["<EXP>"]
    hole_id = corpus.dictionary.word2idx["<HOLE>"]
    l_par_id = corpus.dictionary.word2idx["("]
    r_par_id = corpus.dictionary.word2idx[")"]
except:
    print("EXP, parent. or HOLE index not fount. Please, ensure this is correct behavior")
try:
    pad_id = corpus.dictionary.word2idx["<PAD>"]
except:
    print("PAD index not fount. Please, ensure this is correct behavior")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz, masked):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz if not masked else data[0].size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    if not masked:
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        data.to(device)
    elif masked:
        data = tuple(d.narrow(0, 0, nbatch * bsz) for d in data)
        # Evenly divide the data across the bsz batches.
        data = tuple(d.view(bsz, -1).t().contiguous() for d in data)
        # d.view(bsz, -1, 4).transpose(0,1).contiguous in case of enriched tensor
        data = tuple(d.to(device) for d in data)
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args.masked)
val_data = batchify(corpus.valid, eval_batch_size, args.masked)
test_data = batchify(corpus.test, eval_batch_size, args.masked)



###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    assert not args.masked, "For now no prediction masked goal for transformer"
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    bidirectional = args.masked and (not args.not_bidirectional) 
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, bidirectional=bidirectional).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, masked, ared = False):
    if not masked:
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
    elif masked and not ared:
        source_m, source_l = source
        seq_len = min(args.bptt, len(source_m) - 1 - i)
        data = source_m[i:i+seq_len]
        target = source_l[i:i+seq_len].view(-1)
    elif masked and ared:
        assert False, "Not now"
        source_m, source_l, source_enr = source
        seq_len = min(args.bptt, len(source_m) - 1 - i)
        data = source_m[i:i+seq_len]
        sl = source_enr[i:i+seq_len].tolist()
        stl = source_enr[i:i+seq_len].transpose(0,1).tolist()
        counts=[0 for row in stl]
        for i,row in enumerate(stl):
            for col in row:
                counts[i] += (arity:=col[3])*(n_tokens_to_add:=3)
        max_padding = max(counts)
        data_padded = torch.nn.functional.pad(data, (0,0,0,max_padding))

        lx=source.transpose(0,2).tolist()
        
        lx0_copy = []
        for a,c in enumerate(lx[0]):
            new_c = []
            for b,e in enumerate(c):
                new_c.append(e)
                if e == mask_index:
                    arity = lx[3][a][b]
                    label_id = lx[2][a][b]
                    new_c.extend([label_id]+[l_par_id, hole_id, r_par_id]*arity+[pad_id]*(max_padding-3*arity))
            lx0_copy.append(new_c)
        padded_target = torch.tensor(lx0_copy).transpose(0,2)
        target = padded_target.view(-1)
        data = data_padded
    return data, target


def evaluate(data_source, masked):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        data_source_size = data_source.size(0) - 1 if not masked else data_source[0].size(0) - 1
        for i in range(0, data_source_size, args.bptt):
            data, targets = get_batch(data_source, i, masked)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (data_source_size - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    train_data_size = train_data.size(0) if not args.masked else train_data[0].size(0)
    for batch, i in tqdm.tqdm(enumerate(range(0, train_data_size - 1, args.bptt)), total = train_data_size//args.bptt):
        data, targets = get_batch(train_data, i, args.masked)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            len_train_data = len(train_data) if not args.masked else len(train_data[0])
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len_train_data // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in tqdm.tqdm(range(1, args.epochs+1)):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, args.masked)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data, args.masked)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

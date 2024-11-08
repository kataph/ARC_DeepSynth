import math
import torch
import time
import tqdm
import cProfile

cuda = False
mps = False
batch_size = 5
batch_number = 10
learning_rate = 0.001 #20???
code_model_name = "DSL\\model-LSTM-converged-13kmskdprgms.pt"
vision_model_name = "openai/clip-vit-base-patch32"
clip = 0.25
log_interval = 1
savefile = './model.pt'

if torch.cuda.is_available():
    if not cuda == "cuda":
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = mps and torch.backends.mps.is_available()
if cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


###############################################################################
# Load data
###############################################################################

from write_random_ARC_walks import ARCWalksDataset
from torch.utils.data import DataLoader
walks_ds = ARCWalksDataset(masked=True)
walks_dl = DataLoader(walks_ds, batch_size=batch_size)
import ARC_constants
# ARC_constants_names = [item for item in dir(ARC_constants) if not item.startswith("__")]
from ARC_formatted_dsl import primitive_types
# tokens are either functions or constants (in primitive types), or belong to {"(", ")", "<PAD>", "<HOLE>", "<EXP>", "var0"}
n_tokens = len(primitive_types) + len({"(" ,")", "<PAD>", "<HOLE>", "<EXP>", "<eos>", "var0"})



###############################################################################
# Build the model
###############################################################################


from model import ARCVisionGrammarAlignment, ARCVisionGrammarPrediction, RNNEncoder, VisionEncoder

code_encoder = RNNEncoder(base_model_name=code_model_name, device=device, receives_batch_first=True)
# after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
#     # Currently, only rnn model supports flatten_parameters function.
#     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
#         model.rnn.flatten_parameters()
code_encoder.base_model.rnn.flatten_parameters() #???
vision_encoder = VisionEncoder(base_model_name=vision_model_name, device=device, receives_batch_first=True)
vision_grammar_model = ARCVisionGrammarPrediction(
    codeEncoder=code_encoder, visionEncoder=vision_encoder, 
    #temperature=0, 
    n_tokens=n_tokens,
    codeEncoderUnfrozen=True, visionEncoderUnfrozen=False # False means that it is frozen and wont update weights
)
criterion = torch.nn.NLLLoss()

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


def train():
    # Turn on training mode which enables dropout.
    vision_grammar_model.train()
    total_loss = 0.
    start_time = time.time()
    
    optimizer = torch.optim.Adam(vision_grammar_model.parameters(), lr=learning_rate)

    # if args.model != 'Transformer':
    hidden = code_encoder.base_model.init_hidden(batch_size)
    
    train_data_size = batch_size * batch_number
    
    for batch_index in tqdm.tqdm(range(1,batch_number+1)):
        batch = next(iter(walks_dl))
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        # if args.model == 'Transformer':
        #     output = model(data)
        #     output = output.view(-1, ntokens)
        # else:
        hidden = repackage_hidden(hidden)

        # returns F.log_softmax(predictions_flattened, dim=1), hidden
        log_softmax, hidden = vision_grammar_model({'image': batch["im1"], 'input_ids': batch["pr_str_masked1"]}, hidden)
        target = batch["pr_str1"].view(-1) #B.S> B*S
        loss = criterion(log_softmax, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        log_softmax, hidden = vision_grammar_model({'image': batch["im2"], 'input_ids': batch["pr_str_masked2"]}, hidden)
        target = batch["pr_str2"].view(-1) #B.S> B*S
        loss = criterion(log_softmax, target)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(vision_grammar_model.parameters(), clip) #TODO: ????
        # for p in vision_grammar_model.parameters():  # TypeError: add_(): argument 'other' (position 1) must be Tensor, not NoneType
        #     p.data.add_(p.grad, alpha=-learning_rate)

        total_loss += loss.item()
        total_loss /= 2

        if batch_index % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | learning_rate {:5.5f} | ms/batch {:5.2f} | '
                    'avg loss {:5.2f} | ppl (=exp avg loss) {:8.2f}'.format(
                batch_index, batch_number, learning_rate, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return loss.item()

# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    time0 = time.time()
    # print("running cprofile...")
    # cProfile.run("train()", 'perf.txt')
    # print("... ended")
    last_loss = train()
    print('-' * 89)
    print('| end of training | time: {:5.2f}s | last loss {:5.2f} | exp last loss ppl {:8.2f}'.format((time.time() - time0), last_loss, math.exp(last_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    with open(savefile, 'wb') as f:
        torch.save(vision_grammar_model, f)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')




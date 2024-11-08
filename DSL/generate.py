###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--masked', action='store_true', 
                    help='masked training objective. Requires dataset composed of (masked_sentence,label) couples')
parser.add_argument('--no-masked', dest='masked', action='store_false')
parser.set_defaults(masked=False)
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")
        
use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data, args.masked)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(bsz = 1)
inp = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        if args.masked:
            while True:
                try:
                    masked_sentence = input("Write here masked sequence: ")
                    tokenized_sentence = corpus.tokenize_input_sentence(masked_sentence)
                    batched_sentence = torch.unsqueeze(tokenized_sentence, dim = 1) # S>S.1
                    output, hidden = model(batched_sentence, hidden) # inp: S.1, output: S.1.Tn
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idxs = word_weights.argmax(dim=1)
                    words = [corpus.dictionary.idx2word[word_idx] for word_idx in word_idxs]
                    
                    words = ' '.join(words)
                    print('Prediction is (recall that RNNs will preserve hidden state between multiple calls):')
                    print(words)
                    outf.write(words)

                    mask_index =  masked_sentence.split().index("<EXP>")
                    predictions_for_mask = word_weights[mask_index].tolist()
                    predictions_for_mask = [(corpus.dictionary.idx2word[i],p) for i,p in enumerate(predictions_for_mask)]
                    predictions_for_mask.sort(key=lambda el:el[1], reverse=True)
                    print("Top five predictions were: ", predictions_for_mask[:5])
                except KeyError as e:
                    print(f"Key error for key '{e}', rewrite sentence")
                except ValueError as e:
                    print(f"Value error for value '{e}', rewrite sentence")
                except KeyboardInterrupt:
                    print("exiting gracefully...")
                    break
                if not is_transformer_model: # reset hidden state
                    hidden = model.init_hidden(bsz = 1)
        elif not args.masked:
            for i in range(args.words):
                if is_transformer_model:
                    output = model(inp, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    inp = torch.cat([inp, word_tensor], 0)
                else:
                    output, hidden = model(inp, hidden) # inp: S.B = 1.1, output: S.B.Tn = 1.1.Tn
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    inp.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))

import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, masked: bool = False):
        self.dictionary = Dictionary()
        if masked: 
            self.tokenize_f = self.tokenize_masked
        else:
            self.tokenize_f = self.tokenize
        if not path == None:
            self.train = self.tokenize_f(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize_f(os.path.join(path, 'valid.txt'))
            self.test  = self.tokenize_f(os.path.join(path, 'test.txt' ))


    def tokenize_input_sentence(self, sentence: str, max_length = None):
        """Tokenizes the given input sentence.
        max_length is optional. If set, it will pad all sentences up to max_length and cut longer ones"""
        words = sentence.split() + ['<eos>']
        if max_length != None:
            if len(words) < max_length:
                words.extend(['<PAD>']*(max_length-len(words)))
            if len(words) > max_length:
                words = words[:-(len(words) - max_length)]
        ids = []
        for word in words:
            self.dictionary.add_word(word)
            ids.append(self.dictionary.word2idx[word])
        return torch.tensor(ids).type(torch.int64)
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
    
    def tokenize_masked(self, path, masked_token = "<EXP>"):
        """Tokenizes a text file containing masked/original couples.
        file line = (masked sequence, mask value) -> (masked tensor, unmasked tensor)"""
        assert os.path.exists(path)
        # Add words to the dictionary
        arity = 0
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                # masked_sentence, label = eval(line.) # for eg sopra la <EXP> la, panca
                words=[]
                try:
                    tup = eval(line.strip())
                    if len(tup) == 2:
                        masked_sentence, label = tup
                    elif len(tup) == 3:
                        masked_sentence, label, _ = tup
                        words.append("<PAD>")
                    else: raise TypeError("Cannot parse masked data, too many elements")
                except: # for eg ('... <EXP> ...', label) or ('... <EXP> ...', label, arity)
                    splitted_line = line.strip().split(',')
                    masked_sentence, label = ",".join(splitted_line[:-1]), splitted_line[-1].strip()
                words.extend(masked_sentence.split() + [label, '<HOLE>', '<eos>'])
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss_m = []
            idss_l = []
            # idss_enr = []
            for line in f:
                try:
                    tup = eval(line.strip())
                    if len(tup) == 2:
                        masked_sentence, label = tup
                    elif len(tup) == 3:
                        masked_sentence, label, arity = tup
                    else: raise TypeError("Cannot parse masked data, too many elements")
                except: # for eg ('... <EXP> ...', label) or ('... <EXP> ...', label, arity)
                    splitted_line = line.strip().split(',')
                    masked_sentence, label = ",".join(splitted_line[:-1]), splitted_line[-1].strip()
                    arity = 0
                words_masked = masked_sentence.split() + ['<eos>']
                words_labeled = masked_sentence.replace(masked_token, label).split() + ['<eos>']
                # words_labeled = masked_sentence.replace(masked_token, label+' ( <HOLE> )'*arity).split() + ['<eos>']
                ids_masked = [self.dictionary.word2idx[word_m] for word_m in words_masked]
                ids_labeled = [self.dictionary.word2idx[word_l] for word_l in words_labeled]
                # ids_enriched = [[self.dictionary.word2idx[word_m], 0, 0, 0] if word_m != masked_token else [self.dictionary.word2idx[word_m], 1, self.dictionary.word2idx[label], arity] for word_m in words_masked]
                idss_m.append(torch.tensor(ids_masked).type(torch.int64))
                idss_l.append(torch.tensor(ids_labeled).type(torch.int64))
                # idss_enr.append(torch.tensor(ids_enriched).type(torch.int64))
            ids_m, ids_l = torch.cat(idss_m), torch.cat(idss_l)
            # ids_enr = torch.cat(idss_enr)
            assert ids_m.shape == ids_l.shape
        # return ids_enr #
        return ids_m, ids_l

if __name__ == "__main__":
    capraCorpus = Corpus("data\\smalltext_masked\\", masked=True)
    print(capraCorpus.dictionary.idx2word)
    print(capraCorpus.dictionary.word2idx)
    print(capraCorpus.test)
    prCorpus = Corpus("data\\ARC_masked_programs_ared\\", masked = True)
    print(prCorpus.dictionary.idx2word)
    print(prCorpus.dictionary.word2idx)
    print(prCorpus.test)
    prCorpus = Corpus("data\\ARC_masked_programs\\", masked = True)
    print(prCorpus.dictionary.idx2word)
    print(prCorpus.dictionary.word2idx)
    print(prCorpus.test)
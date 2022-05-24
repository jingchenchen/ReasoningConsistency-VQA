def tokenize(sentence,
                 ignoredPunct=["?", "!", "\\", "/", ")", "("],
                 keptPunct=[".", ",", ";", ":"]):
    sentence = sentence.lower()
    for punct in keptPunct:
        sentence = sentence.replace(punct, " " + punct + " ")
    for punct in ignoredPunct:
        sentence = sentence.replace(punct, "")
    tokens = sentence.split()
    return tokens

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (
            self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict
            else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does'
                             ' not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds

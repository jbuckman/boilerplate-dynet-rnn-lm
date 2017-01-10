from collections import defaultdict
import numpy as np
import math, ast, os, codecs
import cPickle as pickle
import json, sys, io

flatten = lambda l:[item for sublist in l for item in sublist]

recursive_flatten = lambda l:flatten([recursive_flatten(item) if isinstance(item, list) else [item] for item in l])

def normalize(x):
    denom = sum(x)
    return [i/denom for i in x]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def weightedChoice(weights, objects, apply_softmax=False, alpha=None):
    """Return a random item from objects, with the weighting defined by weights
    (which must sum to 1)."""
    if apply_softmax: weights = softmax(weights)
    if alpha: weights = normalize([w**alpha for w in weights])
    cs = np.cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < np.random.rand()) #Find the index of the first weight over a random value.
    idx = min(idx, len(objects)-1)
    return objects[idx]

def itersubclasses(cls, _seen=None):
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

class key_dependent_dict(defaultdict):
    def __init__(self,f_of_x):
        super(key_dependent_dict, self).__init__(None) # base class doesn't get a factory
        self.f_of_x = f_of_x # save f(x)
    def __missing__(self, key): # called when a default needed
        ret = self.f_of_x(key) # calculate default value
        self[key] = ret # and install it in the dict
        return ret

class Token(object):
    def __init__(self, i, s, count=1):
        self.i = i
        self.s = s
        self.count = count
        self.original_s = s

    def __eq__(self, other):
        return self.i == other or self.s == other or \
               (isinstance(other, Token) and self.i == other.i and self.s == other.s)

    def __str__(self): return unicode(self.s)
    def __repr__(self): return str((self.s, self.i))
    def __hash__(self): return self.i

    @staticmethod
    def not_found(): raise Exception("token not found")

class Vocab(object):
    def __init__(self):
        self.tokens = set([])
        self.strings = set([])
        self.s2t = defaultdict(Token.not_found)
        self.i2t = defaultdict(Token.not_found)
        self.unk = None
        self.START_TOK = None
        self.END_TOK = None
        self.delimiter = u''

    @property
    def size(self):
        return len(self.strings)

    def add(self, thing):
        if isinstance(thing, Token): self.add_token(thing)
        else: self.add_string(thing)

    def add_string(self, string):
        if string in self.strings:
            self[string].count += 1
            return self[string]
        i = len(self.tokens)
        s = string
        t = Token(i, s)
        self.i2t[i] = t
        self.s2t[s] = t
        self.tokens.add(t)
        self.strings.add(s)
        return t

    def add_token(self, tok):
        self.i2t[tok.i] = tok
        self.s2t[tok.s] = tok
        self.tokens.add(tok)
        self.strings.add(tok.s)
        return tok

    def __getitem__(self, key):
        if isinstance(key, int): return self.i2t[key]
        elif isinstance(key, Token): return key
        else: return self.s2t[key]

    def unkify(self, string):
        tok = self.unk
        tok.original = string
        return tok

    def add_unk(self, thresh=0, unk_string='<UNK>'):
        if unk_string in self.s2t.keys(): raise Exception("tried to add an UNK token that already existed")
        if self.unk is not None: raise Exception("already added an UNK token")
        strings = [unk_string]
        for token in self.tokens:
            if token.count >= thresh: strings.append(token.s)
        if self.START_TOK is not None and self.START_TOK not in strings: strings.append(self.START_TOK.s)
        if self.END_TOK is not None and self.END_TOK not in strings: strings.append(self.END_TOK.s)
        self.tokens = set([])
        self.strings = set([])
        self.i2t = key_dependent_dict(lambda s:self.unkify(s))
        self.s2t = key_dependent_dict(lambda s:self.unkify(s))
        for string in strings:
            self.add_string(string)
        self.unk = self.s2t[unk_string]
        if self.START_TOK is not None: self.START_TOK = self.s2t[self.START_TOK.s]
        if self.END_TOK is not None: self.END_TOK = self.s2t[self.END_TOK.s]

    def pp(self, seq, delimiter=None):
        if delimiter is None: delimiter = self.delimiter
        return delimiter.join([unicode(self[item].s) for item in seq])

    def hpp(self, seq, delimiter=u''):
        if delimiter is None: delimiter = self.delimiter
        if isinstance(seq, int): return self.i2t[seq]
        else: return "["+delimiter.join([self.hpp(thing) for thing in seq])+"]"

    def save(self, filename):
        info_dict = {
            "tokens":self.tokens,
            "strings":self.strings,
            "s2t":dict(self.s2t),
            "i2t":dict(self.i2t),
            "unk":self.unk,
            "delimiter":self.delimiter,
            "START_TOK":self.START_TOK,
            "END_TOK":self.END_TOK
        }
        with open(filename, "w") as f: pickle.dump(info_dict, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            info_dict = pickle.load(f)
            v = Vocab()
            v.tokens = info_dict["tokens"]
            v.strings = info_dict["strings"]
            v.unk = info_dict["unk"]
            v.delimiter = info_dict["delimiter"]
            defaultf = (lambda :v.unk) if (v.unk is not None) else Token.not_found
            v.s2t = defaultdict(defaultf, info_dict["s2t"])
            v.i2t = defaultdict(defaultf, info_dict["i2t"])
            v.START_TOK = info_dict["START_TOK"]
            v.END_TOK = info_dict["END_TOK"]
            return v

    @classmethod
    def load_from_corpus(cls, reader, remake=False, src_or_tgt="src"):
        vocab_fname = reader.fname+".vocab."+reader.mode+"."+src_or_tgt
        if not remake and os.path.isfile(vocab_fname):
            return Vocab.load(vocab_fname)
        else:
            v = Vocab()
            count = 0
            for item in reader:
                if reader.seq2seq:
                    if src_or_tgt == "src": toklist = item[0]
                    if src_or_tgt == "tgt": toklist = item[1]
                else:
                    toklist = item
                for token in toklist:
                    v.add(token)
                count += 1
                if count % 100 == 0:
                    print "...", count,
                    sys.stdout.flush()
            v.delimiter = reader.delimiter
            v.START_TOK = v[reader.begin] if reader.begin is not None else None
            v.END_TOK = v[reader.end] if reader.end is not None else None
            print "saving vocab of size", v.size
            v.save(vocab_fname)
            return v


#### reader classes

class CorpusReaderTemplate(object):
    names = {"template",}

def get_reader(name):
    for c in itersubclasses(CorpusReaderTemplate):
        if name in c.names: return c
    raise Exception("no reader found with name: " + name)

class GenericLineCorpusReader(CorpusReaderTemplate):
    names = {"generic_char", "generic_word"}
    def __init__(self, fname, begin=None, end=None, mode="generic_char"):
        self.fname = fname
        self.mode = mode
        self.begin = begin
        self.end = end
        self.seq2seq = False
        self.delimiter = '\032' if mode == "generic_char" else ' '
        if mode == "generic_word":
            from pattern.en import tokenize
            self.tokenize = tokenize

    def __iter__(self):
        if os.path.isdir(self.fname):
            filenames = [os.path.join(self.fname,f) for f in os.listdir(self.fname)]
        else:
            filenames = [self.fname]
        for filename in filenames:
            # with io.open(filename, encoding='utf-8') as f:
            with open(filename) as f:
                doc = f.read()
                if self.mode == "generic_char":
                    for line in doc.split("\n"):
                        if not line: continue
                        yield [self.begin] + list(line) + [self.end]
                elif self.mode == "generic_word":
                    for line in doc.split("\n"):
                        if not line: continue
                        yield [self.begin] + ' '.join(self.tokenize(line)).split(" ") + [self.end]

class PTBCorpusReader(CorpusReaderTemplate):
    names = {"ptb", "ptb_bracketed", "ptb_stripped", "ptb_char_stripped", "ptb_nopos"}
    def __init__(self, fname, begin=None, end=None, mode="ptb_bracketed"):
        self.fname = fname
        self.mode = mode
        self.begin = begin
        self.end = end
        self.seq2seq = False
        if "char" in mode:
            self.delimiter = '\032'
        else:
            self.delimiter = ' '

    def __iter__(self):
        if os.path.isdir(self.fname):
            filenames = [os.path.join(self.fname,f) for f in os.listdir(self.fname)]
        else:
            filenames = [self.fname]
        for filename in filenames:
            # with io.open(filename, encoding='utf-8') as f:
            with open(filename) as f:
                doc = f.read()
                for oline in doc.split("\n"):
                    line = oline.strip()
                    if not line: continue
                    line = line.replace(")", " )")
                    line = line.split(" ")
                    if self.mode == "ptb":
                        yield [self.begin] + [item for item in line if item] + [self.end]
                    else:
                        line = [line[i] for i in range(len(line)) if not ((line[i][0] == "(" and i+2 < len(line) and line[i+2][0] == ")")
                                           or (line[i][0] == ")" and i-2 >= 0 and line[i-2][0] == "("))]

                    if self.mode == "ptb_nopos":
                        ans = [item for item in line if item]
                        yield [self.begin] + ans + [self.end]
                    elif self.mode == "ptb_stripped":
                        yield [self.begin] + [item for item in line if item and item[0] not in {"(", ")"}] + [self.end]
                    elif self.mode == "ptb_char_stripped":
                        yield [self.begin] + list(' '.join([item for item in line if item and item[0] not in {"(", ")"}])) + [self.end]
                    elif self.mode == "ptb_bracketed":
                        ans = [self.begin]
                        bracket_stack = []
                        for item in line:
                            if item[0] == "(":
                                ans.append(item)
                                bracket_stack.append(item[1:])
                            elif item[0] == ")":
                                ans.append(item+bracket_stack.pop(-1))
                            else:
                                ans.append(item)
                        ans.append(self.end)
                        yield ans
                    elif self.mode == "ptb_brackets_only":
                        ans = [self.begin]
                        bracket_stack = []
                        for item in line:
                            if item[0] == "(":
                                ans.append(item)
                                bracket_stack.append(item[1:])
                            elif item[0] == ")":
                                ans.append(item+bracket_stack.pop(-1))
                            else:
                                ans.append("<WORD>")
                        ans.append(self.end)
                        yield ans
                    elif self.mode == "raw":
                        yield oline
                    else:
                        raise Exception("bad mode:" + self.mode)
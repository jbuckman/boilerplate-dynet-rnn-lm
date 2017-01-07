import dynet
import random
import util
import os
import pickle
import math
import numpy
from scipy.misc import logsumexp
###########################################################################
class SaveableModel(object):
    name = "template"
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.add_params()

    def add_params(self):
        pass

    def sample(self, first=0, stop=-1, nchars=100):
        pass

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.model.save(path + "/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @staticmethod
    def load(model, path, load_model_params=True):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        lm = get_model(args.mode)(model, args)
        if load_model_params: lm.model.load(path+"/params")
        return lm

class SaveableRNNLM(SaveableModel):
    name = "rnnlm_template"
    def __init__(self, model, vocab, args):
        self.model = model
        self.vocab = vocab
        self.args = args
        self.add_params()

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.vocab.save(path+"/vocab")
        self.model.save(path + "/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @staticmethod
    def load(model, path, load_model_params=True):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        vocab = util.Vocab.load(path+"/vocab")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        lm = get_model(args.arch)(model, vocab, args)
        if load_model_params: lm.model.load(path+"/params")
        return lm

class SaveableS2S(SaveableModel):
    name = "s2s_template"
    def __init__(self, model, src_vocab, tgt_vocab, args):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args
        self.add_params()

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.src_vocab.save(path+"/src_vocab")
        self.tgt_vocab.save(path+"/tgt_vocab")
        self.model.save(path + "/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @staticmethod
    def load(model, path, load_model_params=True):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        src_vocab = util.Vocab.load(path+"/src_vocab")
        tgt_vocab = util.Vocab.load(path+"/tgt_vocab")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        lm = get_model(args.disc_arch)(model, src_vocab, tgt_vocab, args)
        if load_model_params: lm.model.load(path+"/params")
        return lm

def get_model(name):
    for c in util.itersubclasses(SaveableModel):
        if c.name == name: return c
    raise Exception("no language model found with name: " + name)
##########################################################################

class BaselineGenRNNLM(SaveableRNNLM):
    name = "baseline"

    def add_params(self):
        if self.args.rnn == "lstm": rnn = dynet.LSTMBuilder
        elif self.args.rnn == "gru": rnn = dynet.GRUBuilder
        else: rnn = dynet.SimpleRNNBuilder

        # GENERATIVE MODEL PARAMETERS
        self.gen_lookup = self.model.add_lookup_parameters((self.vocab.size, self.args.gen_input_dim))
        self.gen_rnn = rnn(self.args.gen_layers, self.args.gen_input_dim, self.args.gen_hidden_dim, self.model)
        self.gen_R = self.model.add_parameters((self.vocab.size, self.args.gen_hidden_dim))
        self.gen_bias = self.model.add_parameters((self.vocab.size,))

        # print self.vocab.size, self.args.hidden_dim, self.args.input_dim

    def BuildLMGraph(self, sent, sent_args=None):
        if "skip_renew" not in sent_args: dynet.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in sent_args or sent_args["test"] != True)
        if APPLY_DROPOUT: self.gen_rnn.set_dropout(self.args.dropout)
        else: self.gen_rnn.disable_dropout()

        # GENERATIVE MODEL
        init_state = self.gen_rnn.initial_state()
        R = dynet.parameter(self.gen_R)
        bias = dynet.parameter(self.gen_bias)
        errs = [] # will hold expressions
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            x_t = self.gen_lookup[cw]
            state = state.add_input(x_t)
            # if self.vocab[nw].s[0] in {"(", ")"}: continue
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dynet.dropout(y_t, self.args.dropout)
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        gen_err = dynet.esum(errs)
        return gen_err

    def BuildLMGraph_batch(self, batch, sent_args=None):
        if "skip_renew" not in sent_args: dynet.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in sent_args or sent_args["test"] != True)
        if APPLY_DROPOUT: self.gen_rnn.set_dropout(self.args.dropout)
        else: self.gen_rnn.disable_dropout()

        init_state = self.gen_rnn.initial_state()

        #MASK SENTENCES
        isents = [] # Dimension: maxSentLength * minibatch_size

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size

        #No of words processed in this batch
        maxSentLength = max([len(sent) for sent in batch])

        for sent in batch:
            isents.append([self.vocab[word].i for word in sent] + [self.vocab[self.vocab.END_TOK].i for _ in range(maxSentLength-len(sent))])
            masks.append( [1                  for _    in sent] + [0                                for _ in range(maxSentLength-len(sent))])
        isents = map(list, zip(*isents)) # transposes
        masks = map(list, zip(*masks))

        # print isents
        # print masks

        R = dynet.parameter(self.gen_R)
        bias = dynet.parameter(self.gen_bias)
        errs = [] # will hold expressions
        state = init_state

        for (mask, curr_words, next_words) in zip(masks[1:], isents, isents[1:]):
            x_t = dynet.lookup_batch(self.gen_lookup, curr_words)
            state = state.add_input(x_t)
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dynet.dropout(y_t, self.args.dropout)
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax_batch(r_t, next_words)

            ## mask the loss if at least one sentence is shorter. (sents sorted reverse-length, so it must be bottom)
            if mask[-1] == 0:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), len(mask))
                err = err * mask_expr

            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr

    def sample(self, first=0, stop=-1, nchars=100):
        return None
        first = self.vocab[first].i
        stop = self.vocab[stop].i

        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

# Here's an example of how to implement another model to test.
# This model implements the reused word embeddings from https://arxiv.org/pdf/1611.01462v1.pdf
class ReuseEmbeddingsRNNLM(SaveableRNNLM):
    name = "reuse_emb"

    def add_params(self):
        if self.args.rnn == "lstm": rnn = dynet.LSTMBuilder
        elif self.args.rnn == "gru": rnn = dynet.GRUBuilder
        else: rnn = dynet.SimpleRNNBuilder

        # GENERATIVE MODEL PARAMETERS
        self.gen_lookup = self.model.add_lookup_parameters((self.vocab.size, self.args.gen_input_dim))
        self.gen_rnn = rnn(self.args.gen_layers, self.args.gen_input_dim, self.args.gen_hidden_dim, self.model)
        self.gen_R = self.model.add_parameters((self.args.gen_input_dim, self.args.gen_hidden_dim))
        self.gen_bias = self.model.add_parameters((self.args.gen_input_dim,))

    def BuildLMGraph(self, sent, sent_args=None):
        if "skip_renew" not in sent_args: dynet.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in sent_args or sent_args["test"] != True)
        if APPLY_DROPOUT: self.gen_rnn.set_dropout(self.args.dropout)
        else: self.gen_rnn.disable_dropout()

        # GENERATIVE MODEL
        init_state = self.gen_rnn.initial_state()
        R = dynet.parameter(self.gen_R)
        bias = dynet.parameter(self.gen_bias)
        vocab_basis = dynet.transpose(dynet.concatenate_cols([self.gen_lookup[i] for i in range(self.vocab.size)]))
        errs = [] # will hold expressions
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            x_t = self.gen_lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dynet.dropout(y_t, self.args.dropout)
            r_t = vocab_basis * (bias + (R * y_t))
            err = dynet.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        gen_err = dynet.esum(errs)
        return gen_err

    def BuildLMGraph_batch(self, batch, sent_args=None):
        if "skip_renew" not in sent_args: dynet.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in sent_args or sent_args["test"] != True)
        if APPLY_DROPOUT: self.gen_rnn.set_dropout(self.args.dropout)
        else: self.gen_rnn.disable_dropout()

        init_state = self.gen_rnn.initial_state()

        #MASK SENTENCES
        isents = [] # Dimension: maxSentLength * minibatch_size

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size

        #No of words processed in this batch
        maxSentLength = max([len(sent) for sent in batch])

        for sent in batch:
            isents.append([self.vocab[word].i for word in sent] + [self.vocab[self.vocab.END_TOK].i for _ in range(maxSentLength-len(sent))])
            masks.append( [1                  for _    in sent] + [0                                for _ in range(maxSentLength-len(sent))])
        isents = map(list, zip(*isents)) # transposes
        masks = map(list, zip(*masks))

        R = dynet.parameter(self.gen_R)
        bias = dynet.parameter(self.gen_bias)
        vocab_basis = dynet.transpose(dynet.concatenate_cols([self.gen_lookup[i] for i in range(self.vocab.size)]))
        errs = [] # will hold expressions
        state = init_state

        for (mask, curr_words, next_words) in zip(masks[1:], isents, isents[1:]):
            x_t = dynet.lookup_batch(self.gen_lookup, curr_words)
            state = state.add_input(x_t)
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dynet.dropout(y_t, self.args.dropout)
            r_t = vocab_basis * (bias + (R * y_t))
            err = dynet.pickneglogsoftmax_batch(r_t, next_words)

            ## mask the loss if at least one sentence is shorter. (sents sorted reverse-length, so it must be bottom)
            if mask[-1] == 0:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), len(mask))
                err = err * mask_expr

            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr

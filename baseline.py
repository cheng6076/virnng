import dynet as dy

class Baseline(object):
    def __init__(self, reader, word_dim, pretrained_dim, embedding_file):

        self.model = dy.Model()

        self.i2w = reader.i2w_unked
        self.w2i = reader.w2i_unked

        self.vocab = len(self.i2w)

        self.word_lookup = self.model.add_lookup_parameters((self.vocab, word_dim))
        self.pretrained_lookup = self.model.add_lookup_parameters((self.vocab, pretrained_dim))

        self.load_embeddings(embedding_file)


    def __call__(self, x):
        raise NotImplementedError


    def load_embeddings(self, embedding_file):
        if embedding_file is not None:
            embedding_file_fp = open(embedding_file,'r')
            for line in embedding_file_fp:
                line = line.strip().split(' ')
                word, embedding = line[0], [float(f) for f in line[1:]]
                if self.w2i.has_key(word):
                    wid = self.w2i[word]
                    self.pretrained_lookup.init_row(wid, embedding)
            embedding_file_fp.close()


    def get_tok_embedding(self, tok):
        return dy.concatenate([self.word_lookup[self.w2i[tok]], self.pretrained_lookup[self.w2i[tok]]])


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


class MLPAutoencBaseline(Baseline):
    def __init__(self, reader, word_dim, pretrained_dim, embedding_file):
        Baseline.__init__(self, reader, word_dim, pretrained_dim, embedding_file)
        hidden_dim = (word_dim + pretrained_dim) / 2
        self.pW_mlp = self.model.add_parameters((hidden_dim, word_dim + pretrained_dim))
        self.pb_mlp = self.model.add_parameters((hidden_dim, ))
        self.pW_out = self.model.add_parameters((self.vocab, hidden_dim))
        self.pb_out = self.model.add_parameters((self.vocab, ))

    def load_params(self):
        self.W_mlp = dy.parameter(self.pW_mlp)
        self.b_mlp = dy.parameter(self.pb_mlp)
        self.W_out = dy.parameter(self.pW_out)
        self.b_out = dy.parameter(self.pb_out)

    def __call__(self, toks):
        dy.renew_cg()
        self.load_params()

        tok_embeddings = []
        for tok in toks:
            tok_embeddings.append(self.get_tok_embedding(tok))

        h = dy.rectify(self.W_mlp * dy.average(tok_embeddings) + self.b_mlp)
        log_probs = dy.log_softmax(self.W_out * h + self.b_out)
        selected = []
        for tok in toks:
            selected.append(dy.pick(log_probs, self.w2i[tok]))
   
        return dy.esum(selected)


class RNNAutoencBaseline(Baseline):
    def __init__(self, reader, word_dim, pretrained_dim, lstm_dim, embedding_file):
        Baseline.__init__(self, reader, word_dim, pretrained_dim, embedding_file)
        self.pW_input = self.model.add_parameters((lstm_dim, word_dim + pretrained_dim)) 
        self.RNN = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model) 
        self.pW_out = self.model.add_parameters((self.vocab, lstm_dim))
        self.pb_out = self.model.add_parameters((self.vocab, ))
        self.pempty_emb = self.model.add_parameters((lstm_dim,))

    def load_params(self):
        self.W_input = dy.parameter(self.pW_input)
        self.W_out = dy.parameter(self.pW_out)
        self.b_out = dy.parameter(self.pb_out)
        self.empty_emb = dy.parameter(self.pempty_emb)

    def __call__(self, toks):
        dy.renew_cg()
        self.load_params()

        state = self.RNN.initial_state()
        tok_embeddings = []
        for tok in toks:
            tok_embedding = self.W_input * self.get_tok_embedding(tok)
            tok_embeddings.append(tok_embedding)
        
        input_enc_embeddings = tok_embeddings
        input_dec_embeddings = [self.empty_emb] + tok_embeddings[:-1]
        
        for tok_embedding in input_enc_embeddings:
            state = state.add_input(tok_embedding)

        selected = []
        for tid, tok_embedding in enumerate(input_dec_embeddings):
            state = state.add_input(tok_embedding)
            h = state.output()
            log_probs = dy.log_softmax(self.W_out * h + self.b_out)
            selected.append(dy.pick(log_probs, self.w2i[toks[tid]]))

        return dy.esum(selected)


class LanguageModelBaseline(Baseline):
    def __init__(self, reader, word_dim, pretrained_dim, lstm_dim, embedding_file):
        Baseline.__init__(self, reader, word_dim, pretrained_dim, embedding_file)

        self.pW_input = self.model.add_parameters((lstm_dim, word_dim + pretrained_dim)) 
        self.RNN = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model) 
        self.pW_out = self.model.add_parameters((self.vocab, lstm_dim))
        self.pb_out = self.model.add_parameters((self.vocab, ))
        self.pempty_emb = self.model.add_parameters((lstm_dim,))


    def load_params(self):
        self.W_input = dy.parameter(self.pW_input)
        self.W_out = dy.parameter(self.pW_out)
        self.b_out = dy.parameter(self.pb_out)
        self.empty_emb = dy.parameter(self.pempty_emb)


    def __call__(self, toks):
        dy.renew_cg()
        self.load_params()
        state = self.RNN.initial_state()
        tok_embeddings = []
        for tok in toks:
            tok_embeddings.append(self.W_input * self.get_tok_embedding(tok))
        tok_embeddings = [self.empty_emb] + tok_embeddings[:-1]

        selected = []
        for tid, tok_embedding in enumerate(tok_embeddings):
            state = state.add_input(tok_embedding)
            h = state.output()
            log_probs = dy.log_softmax(self.W_out * h + self.b_out)
            selected.append(dy.pick(log_probs, self.w2i[toks[tid]]))

        return dy.esum(selected)


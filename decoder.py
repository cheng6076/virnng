import dynet as dy
import numpy as np

class Decoder:
    def __init__(self, reader, nlayers, word_dim, pretrained_dim, action_dim, lstm_dim, embedding_file):

        self.model = dy.Model()

        self.i2w_word = reader.i2w_unked
        self.w2i_word= reader.w2i_unked
        self.i2w_act = reader.i2w_act
        self.w2i_act = reader.w2i_act
        self.i2w_nt = {}
        self.w2i_nt = {}

        self.NT, self.REDUCE, self.SHIFT = [], self.w2i_act['REDUCE'], self.w2i_act['SHIFT']
        for action in self.w2i_act.keys():
            if 'NT' in action:
                self.NT.append(self.w2i_act[action])
                self.w2i_nt[action] = len(self.w2i_nt)
                self.i2w_nt[len(self.i2w_nt)] = action

        vocab_nt = len(self.i2w_nt)
        vocab_actions = len(self.i2w_act)
        vocab_word = len(self.i2w_word)

        self.pW_input = self.model.add_parameters((lstm_dim, pretrained_dim+ word_dim))    
        self.pW_input_act = self.model.add_parameters((lstm_dim, action_dim))    
        self.pW_input_nt = self.model.add_parameters((lstm_dim, word_dim)) 
        self.pW_input_composed = self.model.add_parameters((lstm_dim, 2 * lstm_dim))
        self.pW_input_ter = self.model.add_parameters((lstm_dim, pretrained_dim + word_dim))    

        self.pW_mlp = self.model.add_parameters((lstm_dim, lstm_dim * 4))
        self.pb_mlp = self.model.add_parameters((lstm_dim, ))

        self.pW_act = self.model.add_parameters((vocab_actions, lstm_dim))
        self.pb_act = self.model.add_parameters((vocab_actions, ))
        self.pW_word = self.model.add_parameters((vocab_word, lstm_dim))
        self.pb_word = self.model.add_parameters((vocab_word, ))

        self.stackRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 
        self.buffRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 
        self.actRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 

        self.word_lookup = self.model.add_lookup_parameters((vocab_word, word_dim))
        self.pretrained_lookup = self.model.add_lookup_parameters((vocab_word, pretrained_dim))
        self.act_lookup = self.model.add_lookup_parameters((vocab_actions, action_dim))
        self.nt_lookup = self.model.add_lookup_parameters((vocab_nt, word_dim))

        self.pempty_buffer_emb = self.model.add_parameters((lstm_dim, ))
        self.pzero_composed_emb = self.model.add_parameters((lstm_dim, ))

        self.load_embeddings(embedding_file)


    def load_embeddings(self, embedding_file):
        """load pretrained embeddings"""
        if embedding_file is not None:
            embedding_file_fp = open(embedding_file,'r')
            for line in embedding_file_fp:
                line = line.strip().split(' ')
                word, embedding = line[0], [float(f) for f in line[1:]]
                if self.w2i_word.has_key(word):
                    wid = self.w2i_word[word]
                    self.pretrained_lookup.init_row(wid, embedding)
            embedding_file_fp.close()


    def load_params(self):
        self.empty_buffer_emb = dy.parameter(self.pempty_buffer_emb)
        self.zero_composed_emb = dy.parameter(self.pzero_composed_emb)
        self.W_input = dy.parameter(self.pW_input)
        self.W_input_act = dy.parameter(self.pW_input_act)
        self.W_input_nt = dy.parameter(self.pW_input_nt)
        self.W_input_ter = dy.parameter(self.pW_input_ter)
        self.W_input_composed = dy.parameter(self.pW_input_composed)
        self.W_mlp = dy.parameter(self.pW_mlp)
        self.b_mlp = dy.parameter(self.pb_mlp)
        self.W_act = dy.parameter(self.pW_act)
        self.b_act = dy.parameter(self.pb_act)
        self.W_word = dy.parameter(self.pW_word)
        self.b_word = dy.parameter(self.pb_word)


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def get_tok_embedding(self, tok):
        tok_embedding = dy.concatenate([dy.lookup(self.word_lookup, self.w2i_word[tok]),
                                        dy.lookup(self.pretrained_lookup, self.w2i_word[tok], update=False)])
        return tok_embedding


    def encode_sentence(self, toks):
        """
        To score a sentence or parse tree, we can encode the forward tokens all at once.
        This will not be used in generation mode.
        """
        state_buffer = self.buffRNN.initial_state()

        tok_embeddings = []
        buffer = []

        for tok in toks:
            tok_embeddings.append(self.get_tok_embedding(tok))
            state_buffer = state_buffer.add_input(self.W_input * tok_embeddings[-1])
            buffer.append(state_buffer.output())

        return tok_embeddings, buffer


    def compute_loss(self, toks, oracle_actions, dropout=0):
        dy.renew_cg()
        self.load_params()

        oracle_actions = list(oracle_actions)
        toks = list(toks)

        stack = []
        stack_top = self.stackRNN.initial_state()
        state_buffer = self.buffRNN.initial_state()
        state_act = self.actRNN.initial_state()

        tok_embeddings, buffer = self.encode_sentence(toks)
        toks.reverse()
        buffer.reverse()

        buffer_embedding = self.empty_buffer_emb
        reducable = 0 
        losses_word = []
        losses_action = []

        while not (len(stack) == 1 and len(buffer) == 0):
            # based on parser state, get valid actions
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self.w2i_act['NT(TOP)']]
            if len(buffer) > 0 and len(stack) > 0:
                valid_actions += self.NT
                valid_actions += [self.SHIFT]
            if (len(stack) >= 2 and reducable != 0) or len(buffer) == 0: 
                valid_actions += [self.REDUCE]

            action = self.w2i_act[oracle_actions.pop(0)]
            log_probs = None
            if len(valid_actions) > 1 or (len(stack) > 0 and valid_actions[0] != self.REDUCE):
                stack_embedding = stack[-1][0].output() 
                act_summary = state_act.output()
                for i in range(len(stack)):
                    if stack[len(stack)-1-i][1] == 'p':
                        parent_embedding = stack[len(stack)-1-i][2]
                        break
                parser_state = dy.concatenate([parent_embedding, act_summary, buffer_embedding, stack_embedding])
                h = dy.rectify(self.W_mlp * parser_state + self.b_mlp)
                if dropout > 0:
                    h = dy.dropout(h, dropout)
                log_probs = dy.log_softmax(self.W_act * h + self.b_act, valid_actions)

                if len(valid_actions) > 1:
                    losses_action.append(dy.pick(log_probs, action))

            act_embedding = self.W_input_act * self.act_lookup[action]
            state_act = state_act.add_input(act_embedding)

            # execute the action to update the parser state
            if action == self.SHIFT:
                log_probs_word = dy.log_softmax(self.W_word * h + self.b_word)
                tok = toks.pop()
                losses_word.append(dy.pick(log_probs_word, self.w2i_word[tok]))

                buffer_embedding = buffer.pop()
                tok_embedding = tok_embeddings.pop()
                tok_embedding = self.W_input_ter * tok_embedding
                stack_state, _, _ = stack[-1] if stack else (stack_top, 'r', stack_top)
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, 'c', tok_embedding))

            elif action in self.NT:
                stack_state, _, _ = stack[-1] if stack else (stack_top, 'r', stack_top)
                nt_embedding = self.W_input_nt * self.nt_lookup[self.w2i_nt[self.i2w_act[action]]]
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

            else:
                found_parent = 0
                path_input = []
                composed_rep = []
                while found_parent != 1:
                    top = stack.pop()
                    top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                    path_input.append(top_raw_rep)
                    if top_label == 'p': found_parent = 1

                nt_emb = path_input.pop()
                if len(path_input) > 0:
                    composed_rep = dy.average(path_input)
                else:
                    composed_rep = self.zero_composed_emb

                top_stack_state, _, _ = stack[-1] if stack else (stack_top, 'r', stack_top)
                composed_embedding = dy.rectify(self.W_input_composed * dy.concatenate([composed_rep, nt_emb]))
                top_stack_state = top_stack_state.add_input(composed_embedding)
                stack.append((top_stack_state, 'c', composed_embedding))    

            if stack[-1][1] == 'p':
                reducable = 0
            else:
                count_p = 0
                for item in stack:
                    if item[1] == 'p': count_p += 1
                if not (count_p == 1 and len(buffer)>0) :
                    reducable = 1
                else:
                    reducable = 0
      
        total_loss_act = -dy.esum(losses_action)
        total_loss_word = -dy.esum(losses_word)

        return dy.esum([total_loss_act, total_loss_word]), total_loss_act, total_loss_word 


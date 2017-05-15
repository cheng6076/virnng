from __future__ import print_function
from operator import itemgetter
from collections import defaultdict
import random
import dynet as dy
import numpy as np

class Encoder:
    def __init__(self, reader, nlayers, word_dim, pretrained_dim, pos_dim, action_dim, lstm_dim, embedding_file):

        self.model = dy.Model()

        self.i2w_raw = reader.i2w_raw
        self.w2i_raw = reader.w2i_raw
        self.i2w_pretrained = reader.i2w_pretrained
        self.w2i_pretrained = reader.w2i_pretrained
        self.i2w_unked = reader.i2w_unked
        self.w2i_unked = reader.w2i_unked
        self.i2w_pos = reader.i2w_pos
        self.w2i_pos = reader.w2i_pos
        self.i2w_act = reader.i2w_act
        self.w2i_act = reader.w2i_act
        self.i2w_nt = {}
        self.w2i_nt = {}

        vocab_actions = len(self.i2w_act)
        vocab_raw = len(self.i2w_raw)
        vocab_pretrained = len(self.i2w_pretrained)
        vocab_unked = len(self.i2w_unked)
        vocab_pos = len(self.i2w_pos)
        self.vocab_actions = vocab_actions

        self.NT, self.REDUCE, self.SHIFT = [], self.w2i_act['REDUCE'], self.w2i_act['SHIFT']
        for action in self.w2i_act.keys():
            if 'NT' in action:
                self.NT.append(self.w2i_act[action])
                self.w2i_nt[action] = len(self.w2i_nt)
                self.i2w_nt[len(self.i2w_nt)] = action
        vocab_nt = len(self.i2w_nt)

        self.pW_input = self.model.add_parameters((lstm_dim, word_dim + 2 * pretrained_dim + pos_dim)) 
        self.pW_input_act = self.model.add_parameters((lstm_dim, action_dim))    
        self.pW_input_nt = self.model.add_parameters((lstm_dim, word_dim)) 
        self.pW_input_composed = self.model.add_parameters((lstm_dim, 2 * lstm_dim))
        self.pW_input_ter = self.model.add_parameters((lstm_dim, word_dim + 2 * pretrained_dim + pos_dim))    

        self.pW_mlp = self.model.add_parameters((lstm_dim, lstm_dim * 7))
        self.pb_mlp = self.model.add_parameters((lstm_dim, ))

        self.pW_act = self.model.add_parameters((vocab_actions, lstm_dim))
        self.pb_act = self.model.add_parameters((vocab_actions, ))

        self.stackRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 
        self.forward_buffRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 
        self.backward_buffRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 
        self.actRNN = dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model) 

        self.word_lookup = self.model.add_lookup_parameters((vocab_raw, word_dim))
        self.pretrained_lookup = self.model.add_lookup_parameters((vocab_pretrained, pretrained_dim))
        self.unked_lookup = self.model.add_lookup_parameters((vocab_unked, pretrained_dim))
        self.pos_lookup = self.model.add_lookup_parameters((vocab_pos, pos_dim))
        self.act_lookup = self.model.add_lookup_parameters((vocab_actions, action_dim))
        self.nt_lookup = self.model.add_lookup_parameters((vocab_nt, word_dim))

        self.pempty_buffer_emb = self.model.add_parameters((2 * lstm_dim,))
        self.pzero_composed_emb = self.model.add_parameters((2 * lstm_dim,))

        self.attention_w1 = self.model.add_parameters((lstm_dim, 2 * lstm_dim))
        self.attention_w2 = self.model.add_parameters((lstm_dim, lstm_dim)) 
        self.attention_v = self.model.add_parameters((1, lstm_dim))

        self.load_embeddings(embedding_file)


    def load_embeddings(self, embedding_file):
        if embedding_file is not None:
            embedding_file_fp = open(embedding_file,'r')
            for line in embedding_file_fp:
                line = line.strip().split(' ')
                word, embedding = line[0], [float(f) for f in line[1:]]
                if self.w2i_pretrained.has_key(word):
                    wid = self.w2i_pretrained[word]
                    self.pretrained_lookup.init_row(wid, embedding)
                if self.w2i_unked.has_key(word):
                    wid = self.w2i_unked[word]
                    self.unked_lookup.init_row(wid, embedding)
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


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def attention(self, input_vectors, state):
        w1 = dy.parameter(self.attention_w1)
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        attention_weights = []

        w2dt = w2*state
        for input_vector in input_vectors:
                attention_weight = v*dy.tanh(w1*input_vector + w2dt)
                attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        output_vectors = dy.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors


    def get_tok_embedding(self, tok):
        tok_embedding = dy.concatenate([dy.lookup(self.word_lookup, self.w2i_raw[tok[0]]),
                                            dy.lookup(self.pretrained_lookup, self.w2i_pretrained[tok[1]], update=False),
                                            dy.lookup(self.unked_lookup, self.w2i_unked[tok[2]]),
                                            self.pos_lookup[self.w2i_pos[tok[3]]]])
        return tok_embedding


    def encode_sentence(self, toks):
        state_forward = self.forward_buffRNN.initial_state()
        state_backward = self.backward_buffRNN.initial_state()

        tok_embeddings = []
        buffer_forward = []
        buffer_backward = []
    
        for tok in toks:
            tok_embeddings.append(dy.rectify(self.W_input * self.get_tok_embedding(tok)))

        for tid in range(len(toks)):
            state_forward = state_forward.add_input(tok_embeddings[tid])
            buffer_forward.append(state_forward.output())

            state_backward = state_backward.add_input(tok_embeddings[len(toks)-1-tid])
            buffer_backward.append(state_backward.output())

        buffer = [dy.concatenate([x, y]) for x, y in zip(buffer_forward, reversed(buffer_backward))]

        return tok_embeddings, buffer


    def train(self, raw, pretrained, unked, pos, oracle_actions, dropout=0):
        oracle_actions = list(oracle_actions)
        toks = [[a,b,c,d] for a, b, c, d in zip(raw, pretrained, unked, pos)]
        toks.reverse()

        dy.renew_cg()
        self.load_params()

        stack = []
        stack_top = self.stackRNN.initial_state()
        state_act = self.actRNN.initial_state()

        losses = []
 
        tok_embeddings, buffer = self.encode_sentence(toks)

        reducable = 0 

        while not (len(stack) == 1 and len(buffer) == 0):
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
                buffer_embedding = buffer[-1] if buffer else self.empty_buffer_emb
                adaptive_buffer_embedding = self.attention(buffer, stack_embedding) if buffer else self.empty_buffer_emb
                act_summary = state_act.output()
                for i in range(len(stack)):
                    if stack[len(stack)-1-i][1] == 'p':
                        parent_embedding = stack[len(stack)-1-i][2]
                        break
                parser_state = dy.concatenate([parent_embedding, act_summary, buffer_embedding, stack_embedding, adaptive_buffer_embedding])
                h = dy.rectify(self.W_mlp * parser_state + self.b_mlp)
                if dropout > 0:
                    h = dy.dropout(h, dropout)
                log_probs = dy.log_softmax(self.W_act * h + self.b_act, valid_actions)

                if len(valid_actions) > 1:
                    losses.append(dy.pick(log_probs, action))

            # execute the action to update the parser state
            if action == self.SHIFT:
                tok = toks.pop()
                buffer.pop()
                #tok_embedding = buffer.pop() 
                tok_embedding = tok_embeddings.pop()
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

            act_embedding = self.W_input_act * self.act_lookup[action]
            state_act = state_act.add_input(act_embedding)

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

        return -dy.esum(losses)


    def parse(self, raw, pretrained, unked, pos, sample=False):
        dy.renew_cg()
        self.load_params()

        toks = [[a,b,c,d] for a, b, c, d in zip (raw, pretrained, unked, pos)]
        toks.reverse()

        stack = []
        stack_top = self.stackRNN.initial_state()
        state_act = self.actRNN.initial_state()
        act_sequence = []
        losses = []
        tok_embeddings, buffer = self.encode_sentence(toks)

        reducable = 1 
        nt_allowed = 1
        ter_allowed = 1
        output = ''
        while not (len(stack) == 1 and len(buffer) == 0):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self.w2i_act['NT(TOP)']]
            if len(buffer) > 0 and len(stack) > 0:
                if nt_allowed:
                    valid_actions += self.NT
                if ter_allowed:
                    valid_actions += [self.SHIFT]
            if (len(stack) >= 2 and reducable != 0) or len(buffer) == 0: 
                valid_actions += [self.REDUCE]

            action = valid_actions[0]

            if len(valid_actions) > 1 or (len(stack) > 0 and valid_actions[0] != self.REDUCE):
                stack_embedding = stack[-1][0].output() 
                buffer_embedding = buffer[-1] if buffer else self.empty_buffer_emb
                adaptive_buffer_embedding = self.attention(buffer, stack_embedding) if buffer else self.empty_buffer_emb
                act_summary = state_act.output()
                for i in range(len(stack)):
                    if stack[len(stack)-1-i][1] == 'p':
                        parent_embedding = stack[len(stack)-1-i][2]
                        break
                parser_state = dy.concatenate([parent_embedding, act_summary, buffer_embedding, stack_embedding, adaptive_buffer_embedding])
                h = dy.rectify(self.W_mlp * parser_state + self.b_mlp)
                log_probs = dy.log_softmax(self.W_act * h + self.b_act, valid_actions)

                if sample:
                    probs = np.exp(log_probs.npvalue() * 0.8)
                    probs = list(probs / probs.sum())
                    action = np.random.choice(range(self.vocab_actions), 1, p=probs)[0]
                    assert (action in valid_actions)
                else:
                    action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
                losses.append(dy.pick(log_probs, action))

            # execute the action to update the parser state
            if action == self.SHIFT:
                tok = toks.pop()
                buffer.pop()
                tok_embedding = tok_embeddings.pop()
                stack_state, _, _ = stack[-1] if stack else (stack_top, 'r', stack_top)
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, 'c', tok_embedding))
                output += tok[0] + ' '

            elif action in self.NT:
                stack_state, _, _ = stack[-1] if stack else (stack_top, 'r', stack_top)
                nt_embedding = self.W_input_nt * self.nt_lookup[self.w2i_nt[self.i2w_act[action]]]
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))
                output += self.i2w_act[action].rstrip('\)').lstrip('NT') + ' '

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
                output += ') '

            act_embedding = self.W_input_act * self.act_lookup[action]
            state_act = state_act.add_input(act_embedding)
            act_sequence.append(self.i2w_act[action])

            #red not allowed after an nt, or stack will be empty before buffer
            reducable = 1
            if stack[-1][1] == 'p':
                reducable = 0
            else:
                count_p = 0
                for item in stack:
                    if item[1] == 'p': count_p += 1
                if (count_p == 1 and len(buffer)>0) :
                    reducable = 0

            #nt is disabled if maximum open non-terminal allowed is reached
            nt_allowed = 1
            count_p = 0
            for item in stack[::-1]:
                if item[1] == 'p':
                    count_p += 1
                else:
                    break
            if count_p >= 6:
                nt_allowed = 0

            #ter is disabled if maximum children under the open nt is reached
            ter_allowed = 1
            count_c = 0
            for item in stack[::-1]:
                if item[1] == 'c':
                    count_c += 1
                else:
                    break
            if count_c >= 20:
                ter_allowed = 0

        return -dy.esum(losses), output, act_sequence


from encoder import Encoder
from decoder import Decoder
from parser import Parser
from baseline import *
from language_model import LanguageModel
from util import Reader
import dynet as dy
from misc import compute_eval_score, compute_perplexity

import os

initializers = {'glorot': dy.GlorotInitializer(),
                'constant': dy.ConstInitializer(0.01),
                'uniform': dy.UniformInitializer(0.1),
                'normal': dy.NormalInitializer(mean = 0, var = 1)
               }

optimizers = {
               "sgd": dy.SimpleSGDTrainer,
               "adam": dy.AdamTrainer,
               "adadelta": dy.AdadeltaTrainer,
               "adagrad": dy.AdagradTrainer
              }


class Session(object):
    def __init__(self, options):
        self.reader = Reader(options.data_dir, options.data_augment)
        self.options = options

    def supervised_enc(self):
        encoder = self.create_encoder()
        if os.path.exists(self.options.result_dir + 'model_enc'):
            self.load_encoder(encoder)

        enc_trainer = optimizers[self.options.optimizer](encoder.model) 
        lr = self.options.lr #used only for sgd

        i = 0
        best_f1 = 0
        print ('supervised training for encoder...')
        for epoch in range(self.options.epochs):
            sents = 0
            total_loss = 0.0

            train = self.reader.next_example(0)
            train_size = len(self.reader.data[0])

            for data in train:
                s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
                loss = encoder.train(s1, s2, s3, pos, act, self.options.enc_dropout)
                sents += 1
                if loss is not None:
                    total_loss += loss.scalar_value()
                    loss.backward()
                    if self.options.optimizer == 'sgd':
                        enc_trainer.update(lr)
                    else:
                        enc_trainer.update()

                e = float(i) / train_size
                if i % self.options.print_every == 0:
                    print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                    sents = 0
                    total_loss = 0.0

                if i!=0 and i % self.options.save_every == 0:
                    print('computing loss on validation set...')
                    valid = self.reader.next_example(1) 
                    valid_size = len(self.reader.data[1])
                    rf = open(self.options.result_dir+'result', 'w')
                    for vdata in valid:
                        s1, s2, s3, pos, act = vdata[0], vdata[1], vdata[2], vdata[3], vdata[4]
                        _, output, _ = encoder.parse(s1, s2, s3, pos)
                        rf.write(output + '\n')
                    rf.close()

                    f1 = compute_eval_score(self.options.result_dir)
                    if f1 > best_f1:
                        best_f1 = f1
                        print ('highest f1: {}'.format(f1)) 
                        print ('saving model...')
                        encoder.Save(self.options.result_dir + 'model_enc')
                    else:
                        lr = lr * self.options.decay 
                i += 1


    def supervised_dec(self):
        decoder = self.create_decoder()
        if os.path.exists(self.options.result_dir + 'model_dec'):
            self.load_decoder(decoder)

        dec_trainer = optimizers[self.options.optimizer](decoder.model) 
        lr = self.options.lr #used only for sgd

        i = 0
        lowest_valid_loss = 9999
        print ('supervised training for decoder...')
        for epoch in range(self.options.epochs):
            sents = 0
            total_loss = 0.0

            train = self.reader.next_example(0)
            train_size = len(self.reader.data[0])

            for data in train:
                s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
                loss, loss_act, loss_word = decoder.compute_loss(s3, act, self.options.dec_dropout)
                sents += 1
                if loss is not None:
                  total_loss += loss.scalar_value()
                  loss.backward()
                  if self.options.optimizer == 'sgd':
                      dec_trainer.update(lr)
                  else:
                      dec_trainer.update()

                e = float(i) / train_size
                if i % self.options.print_every == 0:
                    print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                    sents = 0
                    total_loss = 0.0
                if i!=0 and i % self.options.save_every == 0:
                    print('computing loss on validation set...')
                    total_valid_loss = 0
                    valid = self.reader.next_example(1)
                    valid_size = len(self.reader.data[1])
                    for vdata in valid:
                        s1, s2, s3, pos, act = vdata[0], vdata[1], vdata[2], vdata[3], vdata[4]
                        valid_loss, _, _ = decoder.compute_loss(s3, act)
                        if valid_loss is not None: 
                            total_valid_loss += valid_loss.scalar_value()
                    total_valid_loss = total_valid_loss * 1.0 / valid_size
                    if total_valid_loss < lowest_valid_loss:
                        lowest_valid_loss = total_valid_loss
                        print ('saving model...')
                        decoder.Save(self.options.result_dir + 'model_dec')
                    else:
                        lr = lr * self.options.decay 
                i += 1

    
    def unsupervised_with_baseline(self):
        decoder = self.create_decoder()
        assert(os.path.exists(self.options.result_dir + 'model_dec'))
        self.load_decoder(decoder)

        encoder = self.create_encoder()
        assert(os.path.exists(self.options.result_dir + 'model_enc'))
        self.load_encoder(encoder)

        baseline = self.create_baseline()        
        if os.path.exists(self.options.result_dir + 'baseline'):
            self.load_baseline(baseline)

        enc_trainer = optimizers[self.options.optimizer](encoder.model) 
        dec_trainer = optimizers[self.options.optimizer](decoder.model) 
        baseline_trainer = optimizers[self.options.optimizer](baseline.model)
        lr = self.options.lr #used only for sgd

        i = 0
        lowest_valid_loss = 9999
        print ('unsupervised training...')
        for epoch in range(self.options.epochs):
            sents = 0
            total_loss = 0.0

            train = self.reader.next_example(0)
            train_size = len(self.reader.data[0])

            for data in train:
                s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
                sents += 1

                # random sample
                enc_loss_act, _, act = encoder.parse(s1, s2, s3, pos, sample=True)
                _, dec_loss_act, dec_loss_word = decoder.compute_loss(s3, act)

                # save reward
                logpx = -dec_loss_word.scalar_value()
                total_loss -= logpx

                # reconstruction and regularization loss backprop to theta_d
                dec_loss_total = dec_loss_word + dec_loss_act * dy.scalarInput(self.options.dec_reg)
                dec_loss_total = dec_loss_total * dy.scalarInput(1.0 / self.options.mcsamples)
                dec_loss_total.scalar_value()
                dec_loss_total.backward()

                # update decoder
                if self.options.optimizer == 'sgd':
                    dec_trainer.update(lr)
                else:
                    dec_trainer.update()

                # compute baseline and backprop to theta_b
                b = baseline(s3)
                logpxb = b.scalar_value()
                b_loss = dy.squared_distance(b, dy.scalarInput(logpx))
                b_loss.value()
                b_loss.backward()

                # update baseline
                if self.options.optimizer == 'sgd':
                    baseline_trainer.update(lr)
                else:
                    baseline_trainer.update()

                # policy and and regularization loss backprop to theta_e 
                enc_loss_act = encoder.train(s1, s2, s3, pos, act)
                enc_loss_policy = enc_loss_act * dy.scalarInput((logpx - logpxb) / len(s1))
                enc_loss_total = enc_loss_policy - enc_loss_act * dy.scalarInput(self.options.enc_reg)
                enc_loss_total = enc_loss_total * dy.scalarInput(1.0 / self.options.mcsamples)
                enc_loss_total.value()
                enc_loss_total.backward()

                # update encoder
                if self.options.optimizer == 'sgd':
                    enc_trainer.update(lr)
                else:
                    enc_trainer.update()

                e = float(i) / train_size
                if i % self.options.print_every == 0:
                    print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                    sents = 0
                    total_loss = 0.0
                if i!=0 and i % self.options.save_every == 0:
                    print('computing loss on validation set...')
                    total_valid_loss = 0
                    valid = self.reader.next_example(1)
                    valid_size = len(self.reader.data[1])
                    for vdata in valid:
                        s1, s2, s3, pos, act = vdata[0], vdata[1], vdata[2], vdata[3], vdata[4]
                        _, _, valid_word_loss = decoder.compute_loss(s3, act)
                        # this measure may be not correct
                        if valid_word_loss is not None:
                            total_valid_loss += valid_word_loss.scalar_value()
                    total_valid_loss = total_valid_loss * 1.0 / valid_size
                    if total_valid_loss < lowest_valid_loss:
                        lowest_valid_loss = total_valid_loss
                        print ('saving model...')
                        encoder.Save(self.options.result_dir + 'model_enc')
                        decoder.Save(self.options.result_dir + 'model_dec')
                        baseline.Save(self.options.result_dir + 'baseline')
                    else:
                        lr = lr * self.options.decay
                i += 1
              
        
    def unsupervised_without_baseline(self):
        decoder = self.create_decoder()
        assert(os.path.exists(self.options.result_dir + 'model_dec'))
        self.load_decoder(decoder)

        encoder = self.create_encoder()
        assert(os.path.exists(self.options.result_dir + 'model_enc'))
        self.load_encoder(encoder)
 
        enc_trainer = optimizers[self.options.optimizer](encoder.model) 
        dec_trainer = optimizers[self.options.optimizer](decoder.model) 
        lr = self.options.lr #used only for sgd
        
        i = 0
        lowest_valid_loss = 9999
        print ('unsupervised training...')
        for epoch in range(self.options.epochs):
            sents = 0
            total_loss = 0.0

            train = self.reader.next_example(0)
            train_size = len(self.reader.data[0])

            for data in train:
                s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
                sents += 1
                # max sample
                enc_loss_act, _, act = encoder.parse(s1, s2, s3, pos, sample=False)        
                _, dec_loss_act, dec_loss_word = decoder.compute_loss(s3, act) 
                logpxb = -dec_loss_word.scalar_value()
                total_loss -= logpxb

                # random sample
                enc_loss_act, _, act = encoder.parse(s1, s2, s3, pos, sample=True)
                _, dec_loss_act, dec_loss_word = decoder.compute_loss(s3, act)

                # save reward
                logpx = -dec_loss_word.scalar_value() 

                # reconstruction and regularization loss backprop to theta_d
                dec_loss_total = dec_loss_word + dec_loss_act * dy.scalarInput(self.options.dec_reg)
                dec_loss_total = dec_loss_total * dy.scalarInput(1.0 / self.options.mcsamples)
                dec_loss_total.scalar_value()
                dec_loss_total.backward()
 
                # update decoder
                if self.options.optimizer == 'sgd':
                    dec_trainer.update(lr)
                else:
                    dec_trainer.update()

                # policy and and regularization loss backprop to theta_e 
                enc_loss_act = encoder.train(s1, s2, s3, pos, act)
                enc_loss_policy = enc_loss_act * dy.scalarInput((logpx - logpxb) / len(s1))     
                enc_loss_total = enc_loss_policy - enc_loss_act * dy.scalarInput(self.options.enc_reg)
                enc_loss_total = enc_loss_total * dy.scalarInput(1.0 / self.options.mcsamples)
                enc_loss_total.value()
                enc_loss_total.backward()

                if self.options.optimizer == 'sgd':
                    enc_trainer.update(lr)
                else:
                    enc_trainer.update()

                e = float(i) / train_size
                if i % self.options.print_every == 0:
                    print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                    sents = 0
                    total_loss = 0.0
                if i!=0 and i % self.options.save_every == 0:
                    print('computing loss on validation set...')
                    total_valid_loss = 0
                    valid = self.reader.next_example(1)
                    valid_size = len(self.reader.data[1])
                    for vdata in valid:
                        s1, s2, s3, pos, act = vdata[0], vdata[1], vdata[2], vdata[3], vdata[4]
                        _, _, valid_word_loss = decoder.compute_loss(s3, act)
                        if valid_word_loss is not None:
                            total_valid_loss += valid_word_loss.scalar_value()
                    total_valid_loss = total_valid_loss * 1.0 / valid_size
                    if total_valid_loss < lowest_valid_loss:
                        lowest_valid_loss = total_valid_loss
                        print ('saving model...')
                        encoder.Save(self.options.result_dir + 'model_enc')
                        decoder.Save(self.options.result_dir + 'model_dec')
                    else:
                        lr = lr * self.options.decay
                i += 1

    def pretrain_baseline(self):
        baseline = self.create_baseline()
        if os.path.exists(self.options.result_dir + 'baseline'):
            self.load_baseline(baseline)

        baseline_trainer = optimizers[self.options.optimizer](baseline.model)
        lr = self.options.lr #used only for sgd

        i = 0
        lowest_valid_loss = 9999
        print ('train baseline, for simplicity use the same data here')
        for epoch in range(self.options.epochs):
            sents = 0
            total_loss = 0.0

            train = self.reader.next_example(0)
            train_size = len(self.reader.data[0])

            for data in train:
                s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
                sents += 1
                loss = -baseline(s3)
               
                if loss is not None:
                  total_loss += loss.scalar_value()
                  loss.backward()
                  if self.options.optimizer == 'sgd':
                      baseline_trainer.update(lr)
                  else:
                      baseline_trainer.update()
 
                e = float(i) / train_size
                if i % self.options.print_every == 0:
                    print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                    sents = 0
                    total_loss = 0.0
                if i!=0 and i % self.options.save_every == 0:
                    print('computing loss on validation set...')
                    total_valid_loss = 0
                    valid = self.reader.next_example(1)
                    valid_size = len(self.reader.data[1])
                    for vdata in valid:
                        s1, s2, s3, pos, act = vdata[0], vdata[1], vdata[2], vdata[3], vdata[4]
                        valid_loss = -baseline(s3)
                        if valid_loss is not None:
                            total_valid_loss += valid_loss.scalar_value()
                    total_valid_loss = total_valid_loss * 1.0 / valid_size
                    if total_valid_loss < lowest_valid_loss:
                        lowest_valid_loss = total_valid_loss
                        print ('saving model...')
                        baseline.Save(self.options.result_dir + 'baseline')
                    else:
                        lr = lr * self.options.decay
                i += 1


    def parsing(self):
        decoder = self.create_decoder()
        assert(os.path.exists(self.options.result_dir + 'model_dec'))
        self.load_decoder(decoder)

        encoder = self.create_encoder()
        assert(os.path.exists(self.options.result_dir + 'model_enc'))
        self.load_encoder(encoder)

        print('parsing...')

        rf = open(os.path.join(self.options.result_dir, 'result'), 'w')
        test = self.reader.next_example(2)
        p = Parser(encoder, decoder)
        for dataid, data in enumerate(test):
            s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]
            output = p(s1, s2, s3, pos, self.options.nsamples)
            rf.write(output + '\n')
        rf.close()

        f1 = compute_eval_score(self.options.result_dir)
        print('bracket F1 score is {}'.format(f1))


    def language_modeling(self):
        decoder = self.create_decoder()
        assert(os.path.exists(self.options.result_dir + 'model_dec'))
        self.load_decoder(decoder)

        encoder = self.create_encoder()
        assert(os.path.exists(self.options.result_dir + 'model_enc'))
        self.load_encoder(encoder)

        print('computing language model score...')

        test = self.reader.next_example(2)
        lm = LanguageModel(encoder, decoder)

        total_ll = 0
        total_tokens = 0
        for dataid, data in enumerate(test):
            s1, s2, s3, pos, act = data[0], data[1], data[2], data[3], data[4]     
            if len(s1) <= 1:
                continue
            total_ll += lm(s1, s2, s3, pos, self.options.nsamples)
            total_tokens += len(s1)
        perp = compute_perplexity(total_ll, total_tokens)
        print('perplexity: {}'.format(perp))            


    def create_decoder(self):
        return Decoder(self.reader,
                       self.options.nlayers,
                       self.options.word_dim,
                       self.options.pretrained_dim,
                       self.options.action_dim,
                       self.options.dec_lstm_dim,
                       self.options.embedding_file)


    def create_encoder(self):
        return Encoder(self.reader,
                       self.options.nlayers,
                       self.options.word_dim,
                       self.options.pretrained_dim,
                       self.options.pos_dim,
                       self.options.action_dim,
                       self.options.enc_lstm_dim,
                       self.options.embedding_file)


    def create_baseline(self):
        baseline = None
        if self.options.baseline == 'rnnlm':
            baseline = LanguageModelBaseline(self.reader, 
                                         self.options.word_dim, 
                                         self.options.pretrained_dim, 
                                         self.options.dec_lstm_dim, 
                                         self.options.embedding_file) 
        elif self.options.baseline == 'rnnauto':
            baseline = RNNAutoencBaseline(self.reader,
                                         self.options.word_dim,
                                         self.options.pretrained_dim,
                                         self.options.dec_lstm_dim,
                                         self.options.embedding_file)
        elif self.options.baseline == 'mlp':
            baseline = MLPAutoencBaseline(self.reader,
                                         self.options.word_dim,
                                         self.options.pretrained_dim,
                                         self.options.embedding_file)
        else:
            raise NotImplementedError("Baseline Not Implmented")

        return baseline


    def load_decoder(self, decoder):
        decoder.Load(self.options.result_dir + 'model_dec')


    def load_encoder(self, encoder):
        encoder.Load(self.options.result_dir + 'model_enc')


    def load_baseline(self, baseline):
        baseline.Load(self.options.result_dir + 'baseline')

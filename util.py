from collections import Counter, defaultdict
from random import shuffle
import os

class Reader:
    def __init__(self, data_dir, data_augment=False):
        self.w2i_raw = {}    
        self.i2w_raw = {}
        self.w2i_pretrained = {}    
        self.i2w_pretrained = {}
        self.w2i_unked = {}    
        self.i2w_unked = {}
        self.w2i_pos = {}    
        self.i2w_pos = {}
        self.w2i_act = {}
        self.i2w_act = {}
        self.file_src = [os.path.join(data_dir, 'train.oracle'), os.path.join(data_dir, 'valid.oracle'), os.path.join(data_dir, 'test.oracle')]
        self.data = defaultdict(list)
        if data_augment:
            self.construct_vocab_augment()
            self.load_data_augment()
        else:
            self.construct_vocab()
            self.load_data()
        print ('vocab size for sentence is {}'.format(len(self.w2i_raw)))
        print ('vocab size for sentence(pretrained) is {}'.format(len(self.w2i_pretrained)))
        print ('vocab size for sentence(unked) is {}'.format(len(self.w2i_unked)))
        print ('vocab size for pos tag is {}'.format(len(self.w2i_pos)))
        print ('vocab size for actions is {}'.format(len(self.w2i_act)))

    def construct_vocab(self):
        """construct vocabulary for words and pos tags"""
        cnt_pos = {} 
        cnt_raw = {}
        cnt_pretrained = {}
        cnt_unked = {}
        cnt_act = {}
        for f in self.file_src:
            with open(f, 'r') as current_file:
                txt = current_file.read().split('\n\n')
                for example in txt:
                    example = example.split('\n')
                    if len(example)<5: continue
                    pos, sen_raw, sen_pretrained, sen_unked = example[1], example[2], example[3], example[4]
                    act = example[5:]
                    cnt_pos.update(dict(Counter(pos.split(' '))))

                    sen_raw = sen_raw.split(' ')
                    sen_pretrained = sen_pretrained.split(' ')
                    sen_unked = sen_unked.split(' ')
                    assert (len(sen_raw) == len(sen_unked))

                    cnt_raw.update(dict(Counter(sen_raw)))
                    cnt_pretrained.update(dict(Counter(sen_pretrained)))
                    cnt_unked.update(dict(Counter(sen_unked)))
                    cnt_act.update(dict(Counter(act)))

        for kid, key in enumerate(cnt_pos.keys()):
            self.w2i_pos[key] = kid
            self.i2w_pos[kid] = key
        for kid, key in enumerate(cnt_raw.keys()):
            self.w2i_raw[key] = kid
            self.i2w_raw[kid] = key
        for kid, key in enumerate(cnt_pretrained.keys()):
            self.w2i_pretrained[key] = kid
            self.i2w_pretrained[kid] = key
        for kid, key in enumerate(cnt_unked.keys()):
            self.w2i_unked[key] = kid
            self.i2w_unked[kid] = key
        for kid, key in enumerate(cnt_act.keys()):
            self.w2i_act[key] = kid
            self.i2w_act[kid] = key

    def construct_vocab_augment(self):
        cnt_pos = {}
        cnt_raw = {}
        cnt_act = {}
        for f in self.file_src:
            with open(f, 'r') as current_file:
                txt = current_file.read().split('\n\n')
                for example in txt:
                    example = example.split('\n')
                    if len(example)<5: continue
                    pos, sen_raw, sen_pretrained, sen_unked = example[1], example[2], example[3], example[4]
                    act = example[5:]
                    cnt_pos.update(dict(Counter(pos.split(' '))))

                    sen_raw = sen_raw.split(' ')
                    sen_pretrained = sen_pretrained.split(' ')
                    sen_unked = sen_unked.split(' ')
                    assert (len(sen_raw) == len(sen_unked))

                    cnt_raw.update(dict(Counter(sen_raw)))
                    cnt_raw.update(dict(Counter(sen_pretrained)))
                    cnt_raw.update(dict(Counter(sen_unked)))
                    cnt_act.update(dict(Counter(act)))

        for kid, key in enumerate(cnt_pos.keys()):
            self.w2i_pos[key] = kid
            self.i2w_pos[kid] = key
        for kid, key in enumerate(cnt_raw.keys()):
            self.w2i_raw[key] = kid
            self.i2w_raw[kid] = key
        for kid, key in enumerate(cnt_act.keys()):
            self.w2i_act[key] = kid
            self.i2w_act[kid] = key

        self.w2i_pretrained = self.w2i_raw
        self.w2i_unked = self.w2i_raw
        self.i2w_pretrained = self.i2w_raw
        self.i2w_unked = self.i2w_raw
     

    def load_data(self):
        for fid, f in enumerate(self.file_src):
            data = []
            with open(f, 'r') as current_file:
                txt = current_file.read().split('\n\n')
                for example in txt:
                    example = example.split('\n')
                    if len(example)<5: continue
                    pos, sen_raw, sen_pretrained, sen_unked = example[1], example[2], example[3], example[4]
                    act = example[5:]
                    sen_raw = sen_raw.split(' ')
                    sen_pretrained = sen_pretrained.split(' ')
                    sen_unked = sen_unked.split(' ')
                    assert (len(sen_raw) == len(sen_unked))

                    self.data[fid].append([list(sen_raw), list(sen_pretrained), list(sen_unked), list(pos.split(' ')), list(act)])                    


    def load_data_augment(self):
        for fid, f in enumerate(self.file_src):
            data = []
            with open(f, 'r') as current_file:
                txt = current_file.read().split('\n\n')
                for example in txt:
                    example = example.split('\n')
                    if len(example)<5: continue
                    pos, sen_raw, sen_pretrained, sen_unked = example[1], example[2], example[3], example[4]
                    act = example[5:]
                    sen_raw = sen_raw.split(' ')
                    sen_pretrained = sen_pretrained.split(' ')
                    sen_unked = sen_unked.split(' ')
                    assert (len(sen_raw) == len(sen_unked))
                    pos = pos.split(' ')
                    self.data[fid].append([list(sen_unked), list(sen_unked), list(sen_unked), list(pos), list(act)])                    
                    # data augmentation for training
                    if fid == 0:
                        self.data[fid].append([list(sen_raw), list(sen_raw), list(sen_raw), list(pos), list(act)])                    
                        self.data[fid].append([list(sen_pretrained), list(sen_pretrained), list(sen_pretrained), list(pos), list(act)])                    
                        self.data[fid].append([list(sen_raw), list(sen_pretrained), list(sen_unked), list(pos), list(act)])                    
        

    def next_example(self, split):
        x = range(len(self.data[split]))
        if split==0: shuffle(x)     
        for i in x:
            yield self.data[split][i]

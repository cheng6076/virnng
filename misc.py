import os 
import math
import tempfile
import numpy as np

def compute_eval_score(result_dir):
    result_path = os.path.join(result_dir, 'result')
    gen_path = os.path.join(result_dir, 'generate_eval.py')
    command = 'python {} {}'.format(gen_path, result_path)
    os.system(command)

    result_path = os.path.join(result_dir, 'resulteval')
    eval_path = os.path.join(result_dir, 'EVALB/sample/sample.tst')
    command = 'cp {} {}'.format(result_path, eval_path)
    os.system(command)

    evalb_path = os.path.join(result_dir, 'EVALB/evalb')
    param_path = os.path.join(result_dir, 'EVALB/COLLINS.prm')
    gold_path = os.path.join(result_dir, 'EVALB/sample/sample.gld')
    sample_path = os.path.join(result_dir, 'EVALB/sample/sample.tst')
    output_path = 'tmp'
    command = './{} -p {} {} {} > {}'.format(evalb_path, param_path, gold_path, sample_path, output_path)
    os.system(command)

    f1 = 0
    brackstr="Bracketing FMeasure"
    evalfile = open(output_path, 'r')
    for line in evalfile:
        if line[0:len(brackstr)] == brackstr:
            f1 = float(line.rstrip()[-5:]) #I followed line 1099/1231 of RNNG 

    evalfile.close()
    os.remove(output_path)

    return f1


def compute_perplexity(total_ll, total_tokens):
    return math.exp(-total_ll / total_tokens)

def get_sparse_feature(vocab_size, word_id):
    vec = np.zeros(vocab_size)
    vec[word_id] = 1
    return vec




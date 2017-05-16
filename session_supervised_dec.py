from optparse import OptionParser
from session import Session

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--dynet-mem", help="memory allocation")
    # file path
    parser.add_option("--data",      dest="data_dir",       metavar="FILE", default="dataset/", help="dir to store input data")
    parser.add_option("--result",    dest="result_dir",     metavar="FILE", default="result/",  help="dir to store results")
    parser.add_option("--embedding", dest="embedding_file", metavar="FILE", default="dataset/", help="dir to store pretrained embeddings")
    # model params
    parser.add_option("--word_dim",       type="int", dest="word_dim",       default=40)
    parser.add_option("--pretrained_dim", type="int", dest="pretrained_dim", default=50)
    parser.add_option("--pos_dim",        type="int", dest="pos_dim",        default=10)
    parser.add_option("--action_dim",     type="int", dest="action_dim",     default=36)
    parser.add_option("--enc_lstm_dim",   type="int", dest="enc_lstm_dim",   default=128)
    parser.add_option("--dec_lstm_dim",   type="int", dest="dec_lstm_dim",   default=256)
    parser.add_option("--nlayers",        type="int", dest="nlayers",        default=2)
    # training options
    parser.add_option("--epochs",      type="int", dest="epochs",      default=50)
    parser.add_option("--print_every", type="int", dest="print_every", default=500)
    parser.add_option("--save_every",  type="int", dest="save_every",  default=10000)
    parser.add_option("--data_augment", action='store_true',  dest="data_augment", default=False)
    # optimization misc
    parser.add_option("--lr",          type="float", dest="lr",            default=1)
    parser.add_option("--lrdecay",     type="float", dest="decay",         default=0.75)
    parser.add_option("--enc_dropout", type="float", dest="enc_dropout",   default=0.2)
    parser.add_option("--dec_dropout", type="float", dest="dec_dropout",   default=0.2)
    parser.add_option("--initializer", dest="initializer", default="glorot", help="[glorot,constant,uniform,normal]")
    parser.add_option("--optimizer",   dest="optimizer",   default="sgd",   help="[sgd,momentum,adam,adadelta]")
    # test options
    parser.add_option("--nsamples", type="int", dest="nsamples", default=1)

    (options, args) = parser.parse_args()

    s = Session(options)
    s.supervised_dec()

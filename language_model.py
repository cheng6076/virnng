import dynet as dy
import math

class LanguageModel(object):

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


    @staticmethod
    def logsumexp(loga, logb):
        if loga is None: return logb
        if logb is None: return loga
        if loga < logb:
            loga, logb = logb, loga
        return logb + math.log(1 + math.exp(loga - logb))


    def __call__(self, s1, s2, s3, pos, nsamples, kld=True, method='is'):
        score = 0
        if nsamples == 1:
            score = self.max_score(s1, s2, s3, pos, kld=kld)
        else:
            if method == 'is':
                score = self.logsumexp_score(s1, s2, s3, pos, nsamples, kld=kld)
            elif method == 'lb':
                score = self.lower_bound(s1, s2, s3, pos, nsamples, kld=kld)
            elif method == 'pr':
                score = self.sample_from_prior(s3, nsamples)
        return score


    def max_score(self, s1, s2, s3, pos, kld=False):
        example_ll = 0
        enc_act_loss, _, act = self.encoder.parse(s1, s2, s3, pos, sample=False)
        enc_act_ll = -enc_act_loss.scalar_value()
        _, dec_act_loss, word_loss = self.decoder.compute_loss(s3, act)
        dec_act_ll = -dec_act_loss.scalar_value()
        word_ll = -word_loss.scalar_value() 
        example_ll += word_ll
        if kld:
            example_ll += dec_act_ll - enc_act_ll
        return example_ll


    def logsumexp_score(self, s1, s2, s3, pos, nsamples, kld=False):
        """compute log p(x) as log E_q(a|x) p(a,x)/q(a|x)"""
        enc_act_loss, _, act = self.encoder.parse(s1, s2, s3, pos, sample=True)
        enc_act_ll = -enc_act_loss.scalar_value()
        _, dec_act_loss, word_loss = self.decoder.compute_loss(s3, act)        
        dec_act_ll = -dec_act_loss.scalar_value()
        example_ll = -word_loss.scalar_value()
        if kld:
            example_ll += dec_act_ll - enc_act_ll 

        for i in range(nsamples-1):
            enc_act_loss, _, act = self.encoder.parse(s1, s2, s3, pos, sample=True)
            enc_act_ll = -enc_act_loss.scalar_value()
            _, dec_act_loss, word_loss = self.decoder.compute_loss(s3, act)        
            dec_act_ll = -dec_act_loss.scalar_value()
            word_ll = -word_loss.scalar_value()
            if kld:
                word_ll += dec_act_ll - enc_act_ll

            example_ll = self.logsumexp(example_ll, word_ll)
        example_ll = example_ll - math.log(nsamples)

        return example_ll


    def lower_bound(self, s1, s2, s3, pos, nsamples, kld=False):
        """approximate log p(x) as  E_q(a|x) log p(a,x)/q(a|x)"""
        example_ll = 0
        for i in range(nsamples):
            enc_act_loss, _, act = self.encoder.parse(s1, s2, s3, pos, sample=True)
            enc_act_ll = -enc_act_loss.scalar_value()
            _, dec_act_loss, word_loss = self.decoder.compute_loss(s3, act)
            dec_act_ll = -dec_act_loss.scalar_value()
            example_ll += -word_loss.scalar_value()
            if kld:
                example_ll += dec_act_ll - enc_act_ll
        example_ll = example_ll / nsamples

        return example_ll
   

    def sample_from_prior(self, s, nsamples):
        """compute log p(x) as log E_p(a) p(x|a)"""
        word_loss = self.decoder.sample(s)  
        example_ll = -word_loss.scalar_value()      
        for i in range(nsamples-1):
            word_loss = self.decoder.sample(s)  
            word_ll = -word_loss.scalar_value()
            example_ll = self.logsumexp(example_ll, word_ll)
        example_ll = example_ll - math.log(nsamples)
     
        return example_ll



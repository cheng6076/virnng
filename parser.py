import dynet as dy


_lowest_loss = 9999

class Parser:

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


    def __call__(self, s1, s2, s3, pos, nsamples):
        output = None
        if nsamples == 1:
            output = self.recognition(s1, s2, s3, pos)
        else:
            output = self.reranking(s1, s2, s3, pos, nsamples)

        return output


    def recognition(self, s1, s2, s3, pos):
        _, output, _ = self.encoder.parse(s1, s2, s3, pos, sample=False)
        return output


    def reranking(self, s1, s2, s3, pos, nsamples):
        output = None
        lowest_loss = _lowest_loss
        for i in range(nsamples):
            _, tmpoutput, act = self.encoder.parse(s1, s2, s3, pos, sample=True)
            loss, _, _ = self.decoder.compute_loss(s3, act)
            loss = loss.scalar_value()
            if loss < lowest_loss:
                lowest_loss = loss
                output = tmpoutput

        return output




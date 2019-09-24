from torch.autograd import Variable
import torch.nn as nn
import torch


class WordDropout(nn.Module):
    r"""During training, randomly zeroes some of the (entire) words of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to zero are randomized on every forward call.

    Furthermore, the outputs are scaled by a factor of *1/(1-p)* during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.1
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.0, inplace=False, dim=2):
        super(WordDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        # dimension of the word dropout (sequence).
        # e.g. in [time, batch, features], i.e. [T, B, D], word dropout is applied on either all D or none.
        self.dim = dim
        self.inplace = inplace

    def forward(self, input, training=False):
        if self.p == 0 or not training:
            return input

        keep_prob = 1 - self.p
        noise = torch.zeros_like(input.data)
        noise = Variable(torch.sum(noise, dim=self.dim))
        noise.data.bernoulli_( self.p )
        noise = noise.byte()
        noise = noise.unsqueeze(self.dim)

        output = input.masked_fill_(noise, 0.)
        output /= keep_prob
        return torch.mul(output, input)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'

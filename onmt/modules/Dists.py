from torch import nn
from torch.distributions import Distribution
from torch.distributions import Normal as PyNormal


def convert_symmetric_dirichlet_to_logistic_normal(concentration, dim):
    return 0., (1. / concentration) * (1. - 2. / dim) + 1. / (concentration * dim)
    #return 0., 1.


class Normal(Distribution):
    def __init__(self, mean, std):
        self.normal = PyNormal(mean,std)

    def mean(self):
        return self.normal.mean

    def params(self):
        return [self.normal.mean,self.normal.std]

    def sample(self):
        """
        Generates a single sample or single batch of samples if the distribution
        parameters are batched.
        """
        return self.normal.sample()

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution parameters
        are batched.
        """
        return self.normal.sample_n(n)


    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
	"""
        return self.normal.log_prob(value)

    def kl(self, other):
        """ 
        KL-divergence between two Normals: KL[N(u_i, s_i) || N(u_j, s_j)] 
        where params_i = [u_i, s_i] and similarly for j.
        Returns a tensor with the dimensionality of the location variable.
        """
        if not isinstance(other, Normal):
            raise ValueError('Impossible')
        location_i, scale_i = self.params()  # [mean, std]
        location_j, scale_j = other.params()  # [mean, std]
        var_i = scale_i ** 2.
        var_j = scale_j ** 2.
        term1 = 1. / (2. * var_j) * ((location_i - location_j) ** 2. + var_i - var_j)
        term2 = torch.log(scale_j) - torch.log(scale_i)
        return term1 + term2


class LogisticNormal(Distribution):
    def __init__(self, loc, scale, n_samples=100):
        self.normal = Normal(loc,scale)
        self.n_samples = n_samples

    def mean(self):
        samples = self.sample_n(self.n_samples)
        #return self.normal.mean
        return samples.mean(0)

    def params(self):
        """ The distribution parameters (mean,std) """
        return self.normal.params()

    def sample(self):
        """
        Generates a single sample or single batch of samples if the distribution
        parameters are batched.
        """
        return nn.functional.softmax(self.normal.sample(),-1)

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution parameters
        are batched.
        """
        return nn.functional.softmax(self.normal.sample_n(n),-1)


    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def kl(self, other):
        if isinstance(other, LogisticNormal):
            return self.normal.kl(other.normal)
        else:
            raise ValueError('Impossible (LogisticNormal): self %s other %s' % (type(self), type(other)))



class Delta(Distribution):
    r"""
    Creates a Delta distribution parameterized by a location `loc`.

    Example::

        >>> m = Delta(torch.Tensor([0.0]))
        >>> m.sample()  # mean==0
         0.
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): location of the distribution
    """

    def __init__(self, loc):
        self.loc = loc

    def params(self):
        """ The distribution parameters (mean,std) """
        return [self.loc]

    def sample(self):
        return self.loc

    def mean(self):
        return self.loc

    def sample_n(self, n):
        # cleanly expand float or Tensor or Variable parameters
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())
        return expand(self.loc)

    def log_prob(self, value):
        raise Exception('Delta is degenerate.')

    def kl(self, other):
        if isinstance(other, Delta):
            return torch.zeros_like(self.loc)
        else:
            raise ValueError('Impossible (Delta): self %s other %s' % (type(self), type(other)))

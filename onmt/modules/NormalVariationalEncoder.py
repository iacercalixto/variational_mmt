import sys
import torch
import torch.nn as nn
from torch.distributions import Distribution
from onmt.Utils import sequence_mask
from torch.autograd import Variable
import numpy
from onmt.modules.Dists import Delta, Normal, LogisticNormal, convert_symmetric_dirichlet_to_logistic_normal
from onmt.modules import GlobalAttention


class LocationLayer(nn.Module):
    """ Layer that implements a location (i.e. computes a mean). """
    def __init__(self, input_size, hidden_size, output_size=None):
        super(LocationLayer, self).__init__()
        if output_size is None:
            output_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out



class ScaleLayer(nn.Module):
    """ Layer that implements a scale (i.e. computes a variance). """
    def __init__(self, input_size, hidden_size, output_size=None):
        super(ScaleLayer, self).__init__()
        if output_size is None:
            output_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.softplus(self.fc2(out))
        return out



class GlobalInferenceNetwork(nn.Module):
    """ Inference network conditioned on the source sentence representations alone. 

        Z_0|x_1^m ~ N(loc(h), scale(h))

        h = average(enc(x_1^m))
        where enc(x_1^m) is an encoding function and x_1^m is the source sentence    
    """
    def __init__(self, z_dim, input_size, dist_type):
        assert(dist_type in ["normal", "logistic_normal"]), \
                "Distribution not supported: %s"%str(dist_type)

        super(GlobalInferenceNetwork, self).__init__()
        self.location = LocationLayer(input_size, z_dim)
        self.scale = ScaleLayer(input_size, z_dim)
        self.dist_type = dist_type
        self.__input_size = input_size

    def encode_seq(self, seq, seq_lengths):
        """ Encode sequence `seq` using its lengths `seq_lengths`
            to mask paddings and return the average sequence encoding.

            seq  (sequence_length, batch_size, feats):
                Sequence encodings.
            seq_length (batch_size):
                Sequence lengths.
        """
        # mask => [B,n], seq_lengths => [B]
        mask = sequence_mask(seq_lengths)
	# mask => [B,n,1]
        mask = mask.unsqueeze(2)  # Make it broadcastable.
        mask = Variable(mask.type(torch.Tensor), requires_grad=False) # convert to a float variable
        # x => [B,n,d]
        seq = seq.transpose(0,1)
        seq = seq*mask
        # average/sum
        h = seq.sum(1) / mask.sum(1)  # [B,d] / [B,1]
        return h


    def fake_input(self, batch_size, real_input_dim):
        input_size_diff = self.__input_size - real_input_dim
        h0 = Variable(torch.zeros(batch_size, input_size_diff))
        return h0


    def forward(self, x, x_lengths):
        """ x (source_length, batch_size, enc_dim):
                Source encodings.
            x_lengths (batch_size):
                array containing lengths for each source sequence.
        """
        # compute the average source hidden state
        # h => [B, src_feats]
        h = self.encode_seq(x, x_lengths)
        # [B,dz]
        loc = self.location(h)
        scale = self.scale(h)

        # the jth variational factor
        if self.dist_type == "normal":
            return Normal(loc, scale), h
        else:
            return LogisticNormal(loc, scale), h


class GlobalFullInferenceNetwork(GlobalInferenceNetwork):
    """ Inference network conditioned on the source-sentence representations,
        target-sentence representations and image features.
        These image features can be global (vector) or local (3D tensor)

        Z_0|x_1^m, y_1^n, v ~ N(loc(h), scale(h))

        h_1 = average(enc_x(x_1^m))
        h_2 = average(enc_y(y_1^n))
        h_3 = average(enc_v(v))
        h = [h_1; h_2; h_3]

        where:
            enc_x(x_1^m) is a source encoding function and x_1^m is the source sentence
            enc_y(y_1^n) is a target encoding function and y_1^n is the target sentence
            enc_v(v)     is an image encoding function and v     is the image
    """
    def __init__(self, z_dim, input_size, dist_type, image_features_type="global"):
        assert(dist_type in ["normal", "logistic_normal"]), \
                "Distribution not supported: %s"%str(dist_type)
        assert(image_features_type in ["global", "posterior", "local"]), \
                "Image features type not supported: %s"%str(image_features_type)

        super(GlobalFullInferenceNetwork, self).__init__(z_dim, input_size, dist_type)

        self.image_features_type = image_features_type
        if image_features_type == 'local':
            attn_type = 'general'
            coverage_attn = False
            # TODO: remove hardcoded hyperparameters
            #hidden_size = input_size // 4
            hidden_size = 500
            #print("input_size: ", input_size)
            #print("hidden_size: ", hidden_size)

            # linear layer to project local image features into RNN hidden state space
            self.image_proj = nn.Linear(2048, 500)

            # create attention mechanisms for the local image features
            self.src_image_attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type
            )
            self.tgt_image_attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type
            )
            self.src_dropout = nn.Dropout(0.5)
            self.tgt_dropout = nn.Dropout(0.5)


    def forward(self, x, x_lengths, y, y_lengths, v):
        """ x (source_length, batch_size, encx_dim):
                Source encodings.
            x_lengths (batch_size):
                Array containing lengths for each source sequence.
            y (target_length, batch_size, ency_dim):
                Target encodings.
            y_lengths (batch_size):
                Array containing lengths for each target sequence.
            v (batch_size, img_dim) or (batch_size, 7, 7, img_dim):
                Image features (global or local).
        """
        # compute the average source hidden states
        # hx => [B, src_feats], hy => [B, tgt_feats]
        hx = self.encode_seq(x, x_lengths)

        if y is None and y_lengths is None and v is None:
            # fake input has size of everything but x
            htemp = self.fake_input( batch_size=hx.size()[0], real_input_dim=hx.size()[-1] )
        else:
            hy = self.encode_seq(y, y_lengths)

            # if we are using local features, run attention.
            # otherwise, if we are using global/posterior features, just use them as-is.
            if self.image_features_type == 'local':
                v = self.image_proj(v)
                # no need to transpose v since it is already batch-major
                src_attn_output, attn = self.src_image_attn(
                    x.transpose(0, 1).contiguous(), # (batch_size, src_words, src_dim)
                    v                               # (batch_size, 7*7, img_dim)
                )
                imgsrc_output = self.src_dropout(src_attn_output)
                tgt_attn_output, attn = self.tgt_image_attn(
                    y.transpose(0, 1).contiguous(), # (batch_size, tgt_words, tgt_dim)
                    v                               # (batch_size, 7*7, img_dim)
                )
                imgtgt_output = self.tgt_dropout(tgt_attn_output)

                # restore outputs to batch-major
                # img???_output => [batch, 7*7, rnn_size]
                imgsrc_output = imgsrc_output.transpose(0,1)
                imgtgt_output = imgtgt_output.transpose(0,1)
                # get the means over the sequences
                # img???_output => [batch, rnn_size]
                imgsrc_output = imgsrc_output.mean(1)
                imgtgt_output = imgtgt_output.mean(1)
                # hv => [batch, 2*rnn_size]
                hv = torch.cat([imgsrc_output,imgtgt_output],-1) # use output of attention over image features
            else:
                hv = v

            # htemp => [batch, 3*rnn_size]
            htemp = torch.cat([hy,hv],-1)

        # concatenate everything together
        h = torch.cat([hx,htemp],-1)

        # [B,dz]
        loc = self.location(h)
        scale = self.scale(h)
        # the jth variational factor
        if self.dist_type == "normal":
            return Normal(loc, scale), h
        else:
            return LogisticNormal(loc, scale), h


class ImageGlobalInferenceNetwork(GlobalInferenceNetwork):
    """ Image inference network that uses one global latent variable to predict image features.

        V | b_1^t ~ Normal(mu, sigma)
        mu = affine(s)
        sigma = softplus(affine(s))

        s = gated-sum of the average bs
        g = sigmoid(affine(b))  --- gates
    """
    def __init__(self, latent_dim, image_feats_dim, src_encodings_dim,
                 use_source_encodings, dist_type):
        """ latent_dim (int):
              latent variable dimensionality.
            image_feats_dim (int):
              global image features dimensionality.
            src_encodings_dim (int):
              source encodings dimensionality.
            use_source_encodings (boolean):
              whether or not to use source encodings (in addition to global latent variable)
              to predict image features.
            dist_type (str):
              distribution type. Must be one of: 'normal', 'logistic_normal'
        """
        assert(dist_type in ["normal", "logistic_normal"]), \
                "Distribution not supported: %s"%str(dist_type)

        z_dim = image_feats_dim
        input_size = latent_dim+src_encodings_dim if use_source_encodings else latent_dim

        super(ImageGlobalInferenceNetwork, self).__init__(z_dim, input_size, dist_type)

        self.use_source_encodings = use_source_encodings
        # gate used to learn to sum over the target words
        self.gate_affine_transform = nn.Linear(input_size, 1)


    def forward(self, z, x, x_lengths):
        """ z (batch_size, latent_dim):
                Global latent variable.
            x (source_length, batch_size, src_feats):
                Source-language encodings (e.g. output of a bi-rnn)
            x_lengths (batch_size):
                List containing, for each sentence in x, the length of the corresponding entry.
        """
        if self.use_source_encodings:
            # [B, src_feats]
            x = self.encode_seq(x, x_lengths)
            # x_and_z => [B, latent_dim+source_hidden_features]
            x_and_z = torch.cat([x, z], dim=-1)
            # compute gates
            # gs => [B,1]
            gs = torch.sigmoid(self.gate_affine_transform(x_and_z))
        else:
            # gs => [B,1]
            gs = torch.sigmoid(self.gate_affine_transform(z))

        # gate the global latent variable
        # use_source_encodings is True:  gated_bs => [B, latent_dim+source_hidden_features]
        # use_source_encodings is False: gated_bs => [B, latent_dim]
        gated_bs = z*gs

        # concatenate xs and z, in case we are using source encodings
        if self.use_source_encodings:
            gated_bs = torch.cat([gated_bs, x], dim=-1)

        # create a normal and return
        # [B,image_feats_dim]
        loc = self.location(gated_bs)
        scale = self.scale(gated_bs)
        if self.dist_type == "normal":
            return Normal(loc, scale), None
        else:
            return LogisticNormal(loc, scale), None


class ImageTopicInferenceNetwork(ImageGlobalInferenceNetwork):
    """ Image inference network that uses topics.

        V | b_1^t ~ Normal(mu, sigma)
        mu = affine(s)
        sigma = softplus(affine(s))

        s = gated-sum of the average bs
        g = sigmoid(affine(b))  --- gates
    """
    def __init__(self, latent_dim, image_feats_dim, src_encodings_dim, use_source_encodings, dist_type):
        assert(dist_type in ["normal", "logistic_normal"]), \
                "Distribution not supported: %s"%str(dist_type)

        super(ImageTopicInferenceNetwork, self).__init__(latent_dim,
                image_feats_dim, src_encodings_dim,
                use_source_encodings, dist_type)


    def forward(self, bs, tgt_mask, x, x_lengths):
        """ bs (target_length, batch_size, topic_dim):
                Average topic embeddings (per decoder timestep).
            tgt_mask (target_length, batch_size):
                Target-language tokens mask.
            x (source_length, batch_size, src_features):
                Source-language encodings (e.g. output of a bi-rnn)
            x_lengths (batch_size):
                List containing, for each sentence in x, the length of the corresponding entry.
        """
        # bs => [B,target_words,topic_dim]
        bs = bs.transpose(0,1)
        # tgt_mask_ => [B, target_words, 1]
        tgt_mask = tgt_mask.transpose(0,1).unsqueeze(2)
        # bs => [B, target_words, topic_dim]
        bs = bs * tgt_mask

        if self.use_source_encodings:
            # x => [B, src_feats]
            x = self.encode_seq(x, x_lengths)
            # x_ => [B, 1, src_feats]
            x_ = x.unsqueeze(1)
            # x_ => [B,target_words,source_hidden_features]
            x_ = x_.repeat(1,bs.size()[1],1)
            # bs_and_xs => [B,target_words,topic_dim+source_hidden_features]
            bs_and_xs = torch.cat([bs, x_], dim=2)
            # compute gates
            # gs => [B,target_words,1]
            gs = torch.sigmoid(self.gate_affine_transform(bs_and_xs))
        else:
            # gs => [B,target_words,1]
            gs = torch.sigmoid(self.gate_affine_transform(bs))
        
        # gate the average topic embeddings
        # gated_bs => [B, topic_dim, 1]
        gated_bs = torch.bmm(bs.transpose(1,2),gs)
        # gated_bs => [B, topic_dim]
        gated_bs = gated_bs.squeeze(2)

        if self.use_source_encodings:
            gated_bs = torch.cat([gated_bs, x], dim=1)

        # [B,image_feats_dim]
        loc = self.location(gated_bs)
        scale = self.scale(gated_bs)

        # [B, target_words, image_feats_dim]
        loc = self.location(gated_bs)
        scale = self.scale(gated_bs)
        # create a distribution and return
        if self.dist_type == "normal":
            return Normal(loc, scale), None
        else:
            return LogisticNormal(loc, scale), None



class EmbeddingInferenceNetwork(nn.Module):
    """ Embedding inference network.
    """
    def __init__(self, topic_dim, number_of_topics, distribution_type='normal'):
        super(EmbeddingInferenceNetwork, self).__init__()
        assert(distribution_type in ['normal', 'delta']), 'Distribution not supported: %s'%distribution_type

        self.distribution_type = distribution_type
        self.location = nn.Parameter(
                torch.Tensor( 1, number_of_topics, topic_dim ),
                requires_grad=True
        )
        if distribution_type == 'normal':
            self.scale = nn.Parameter(
                    torch.Tensor( 1, number_of_topics, topic_dim ),
                    requires_grad=True
            )
            self.softplus = nn.Softplus()
        else:
            self.scale = Variable(
                    torch.zeros(1),
                    requires_grad=False
            )
        self.topic_dim = topic_dim
        self.number_of_topics = number_of_topics

    def forward(self):
        """ Return distribution over embeddings.
        """
        if self.distribution_type=='normal':
            q = Normal(self.location, self.softplus( self.scale ))
        else:
            q = Delta(self.location)
        return q

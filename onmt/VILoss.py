"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import sys

import onmt
import onmt.io
from onmt.Loss import NMTLossCompute
from onmt.modules.Dists import convert_symmetric_dirichlet_to_logistic_normal
import pdb
from collections import defaultdict


def compute_cosine(pred, obs):
    """ calculate cosine similarity between pred and obs.
        if pred and obs are one dimensional arrays,
            normalise them and compute cosine_similarity.
        if pred and obs are two dimensional arrays,
            assume first dimension is batch and second is feature value.
        if pred and obs are three dimensional arrays,
            assume first dimension is batch, second is feature type, and third is feature value.

        pred        (Tensor):           Tensor with predictions.
        obs         (Tensor):           Tensor with observations.
        Returns     (Tensor):           Cosine similarity between predictions and observations.
    """
    assert(pred.numel() == obs.numel()), \
            'Size of observation and prediction tensors much match. Received: pred %s, obs %s.'%(
                    str(pred.size()), str(obs.size()))

    def normalise(x, dim=1):
        """ compute L2 norm and normalise x """
        norm = torch.sqrt( torch.pow(x,2.).sum(dim) )
        if dim>0:
            x /= norm.unsqueeze(dim)
        return x

    # if we have one-dimensional tensors, compute cosine similarity along first dimension (0).
    # if we have two-dimensional tensors, compute cosine similarity along second dimension (1).
    # if we have three-dimensional tensors, compute cosine similarity along third dimension (2).
    # i.e. first dimension is considered the feature vector (will be reduced to a scalar, the cos.sim.)
    dim = len(pred.size()) - 1
    assert(dim>=0 and dim <=2), \
            'This function only computes cosine similarity between 1D, 2D or 3D tensors! Received dim==%i'%(dim)

    p_norm = normalise(pred, dim=dim)
    v_norm = normalise(obs, dim=dim)
    return torch.nn.functional.cosine_similarity( p_norm, v_norm, dim=dim )


class NMTVIModel1LossCompute(NMTLossCompute):
    """
    NMT VI Model1 Loss Computation.
    """
    def __init__(self, generator, tgt_vocab,
                 normalization="sents",
                 label_smoothing=0.0,
                 use_kl_annealing=False,
                 use_kl_freebits=False,
                 kl_freebits_margin=0.0,
                 kl_annealing_current=0.0,
                 kl_annealing_increment=0.0001,
                 kl_annealing_warmup_steps=1000,
                 image_loss_type='logprob',
                 use_local_image_features=False,
                 two_step_image_prediction=False
        ):
        """
            If two_step_image_prediction is True, always predict image pixels (using a categorical) and
                if image_loss_type is 'logprob' and use_local_image_features is True,  predict local image features using logprob.
                if image_loss_type is 'cosine'  and use_local_image_features is True,  predict local image features using cosine.
                if image_loss_type is 'logprob' and use_local_image_features is False, predict global image features using logprob.
                if image_loss_type is 'cosine'  and use_local_image_features is False, predict global image features using cosine.

            If two_step_image_prediction is False,
                if image_loss_type is 'logprob' and use_local_image_features is True,  predict local image features using logprob.
                if image_loss_type is 'cosine'  and use_local_image_features is True,  predict local image features using cosine.
                if image_loss_type is 'logprob' and use_local_image_features is False, predict global image features using logprob.
                if image_loss_type is 'cosine'  and use_local_image_features is False, predict global image features using cosine.
                if image_loss_type is 'categorical', ignore use_local_image_features and predict local image pixels using categorical.
        """
        self.multimodal_model_type = 'vi-model1'

        super(NMTVIModel1LossCompute, self).__init__(generator, tgt_vocab,
                normalization, label_smoothing)

        # kl annealing parameters
        self.n_model_updates = 0
        self.use_kl_annealing = use_kl_annealing
        if use_kl_annealing:
            self.kl_annealing_current       = kl_annealing_current
            self.kl_annealing_increment     = kl_annealing_increment
            self.kl_annealing_warmup_steps  = kl_annealing_warmup_steps
        else:
            self.kl_annealing_current       = 1.0
            self.kl_annealing_increment     = 0.0
            self.kl_annealing_warmup_steps  = 0

        self.use_kl_freebits = use_kl_freebits
        if use_kl_freebits:
            self.kl_freebits_margin = kl_freebits_margin
        else:
            self.kl_freebits_margin = 0.0

        self.image_loss_type = image_loss_type
        self.use_local_image_features = use_local_image_features
        self.two_step_image_prediction = two_step_image_prediction
        self._statistics = onmt.VIStatistics

        if image_loss_type == 'categorical':
            self.image_loss_criterion = nn.NLLLoss2d()

    def _make_shard_state(self, batch, output, range_, attns):
        # q => [B, z_latent_dim]
        q = attns["z_latent"][0]
        loc, scale = q.params()
        # loc, scale => [1, B, z_latent_dim]
        loc = loc.unsqueeze(0)
        scale = scale.unsqueeze(0)

        # p => [B, z_latent_dim]
        p = attns["p_latent"][0]
        p_loc, p_scale = p.params()
        # p_loc, p_scale => [1, B, z_latent_dim]
        p_loc = p_loc.unsqueeze(0)
        p_scale = p_scale.unsqueeze(0)

        if self.two_step_image_prediction:
            # image features
            # pv => [B, image_features_dimensionality]
            pv = attns["p_global_image_features"][0]
            # pv_loc, pv_scale => [1, B, image_features_dimensionality]
            try:
                pv_loc, pv_scale = pv.params()
                pv_loc = pv_loc.unsqueeze(0)
                pv_scale = pv_scale.unsqueeze(0)
            except:
                pv_loc = pv.unsqueeze(0)
                pv_scale = pv.unsqueeze(0)
                pass

            # img_feats => [B, image_features_dimensionality]
            img_feats = attns["ground_truth_global_image_features"]
            img_feats = torch.stack(img_feats)

            # image pixels/vectors
            # p_image_pixels => [B, 3, size, size]
            p_image_pixels = attns["p_global_image_pixels"][0]
            # p_image_pixels => [1, B, 3, size, size]
            p_image_pixels = p_image_pixels.unsqueeze(0)
            # img_vecs => [B, 3, size, size]
            img_vecs = attns["ground_truth_global_image_pixels"]
            img_vecs = torch.stack(img_vecs)
            
            p_image_pixels_scale  = None

        else:
            pv_loc = None
            pv_scale = None
            p_image_pixels = None
            p_image_pixels_scale  = None
            img_vecs = None
            img_feats = None

            # in case we are predicting image features only
            if self.image_loss_type != 'categorical':
                # pv => [B, image_features_dimensionality]
                pv = attns["p_global_image_features"][0]
                # pv_loc, pv_scale => [1, B, image_features_dimensionality]
                try:
                    pv_loc, pv_scale = pv.params()
                    pv_loc = pv_loc.unsqueeze(0)
                    pv_scale = pv_scale.unsqueeze(0)
                except:
                    pv_loc = pv.unsqueeze(0)
                    pv_scale = pv.unsqueeze(0)
                    pass

           
                # img_feats => [B, image_features_dimensionality]
                img_feats = attns["ground_truth_global_image_features"]
                img_feats = torch.stack(img_feats)
            else:
                # in case we are predicting image pixels and not image features
                # pv => [B, channels, size, size]
                p_image_pixels = attns["p_global_image_pixels"][0]
                # pv => [1, B, channels, size, size]
                p_image_pixels = p_image_pixels.unsqueeze(0)
            
                # img_vecs => [B, channels, size, size]
                img_vecs = attns["ground_truth_global_image_pixels"]
                img_vecs = torch.stack(img_vecs)

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "qz_location": loc,
            "qz_scale": scale,
            "pz_location": p_loc,
            "pz_scale": p_scale,
            "p_global_image_features_location": pv_loc,
            "p_global_image_features_scale": pv_scale,
            "ground_truth_global_image_features": img_feats,
            "p_image_pixels_location": p_image_pixels,
            "p_image_pixels_scale": p_image_pixels_scale,
            "ground_truth_image_pixels": img_vecs,
        }

    def _compute_loss(self, batch, output, target,
            qz_location, qz_scale,
            pz_location, pz_scale,
            p_global_image_features_location=None,
            p_global_image_features_scale=None,
            ground_truth_global_image_features=None,
            p_image_pixels_location=None,
            ground_truth_image_pixels=None,
            p_image_pixels_scale=None,
        ):

        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        # original NMT loss (negative log-likelihood)
        loss = self.criterion(scores, gtruth)
        nmt_loss_data = loss.data.clone()
        #print("nmt_loss_data: ", type(nmt_loss_data))

        def reshape_global_image_features(pred_location, pred_scale, obs):
            """ Reshape global image features and groundtruth Variables/tensors. """
            # ground truth features - just squeeze first dim
            # [1, batch, feats] => [batch, feats]
            obs = obs.squeeze(0)
            pred_location = pred_location.squeeze(0)
            pred_scale = pred_scale.squeeze(0)
            return pred_location, pred_scale, obs

        def reshape_local_image_features(pred_location, pred_scale, obs):
            """ Reshape local image features and groundtruth Variables/tensors. """
            # ground truth features - just squeeze first dim
            # [1, batch, 49, 2048] => [batch, 49, 2048]
            obs = obs.squeeze(0)
            if len(obs.size())==2:
                oldshape_ = obs.size()
                obs = obs.view(oldshape_[0], 49, oldshape_[1]//49)

            shape_ = obs.size()
            # predicted features - squeeze first dim and permute dimensions (in numpy, this is done with transpose)
            # same for location and scale
            # [1, batch, 2048, 7, 7] => [batch, 2048, 7, 7]
            pred_location = pred_location.squeeze(0)
            pred_scale    = pred_scale.squeeze(0)
            # [batch, 2048, 7, 7] => [batch, 7, 7, 2048]
            pred_location = pred_location.permute(0, 2, 3, 1).contiguous()
            pred_scale    = pred_scale.permute(0, 2, 3, 1).contiguous()
            shape_ = pred_location.size()
            #shape_ = pred_scale.size()
            pred_location = pred_location.view(shape_[0], shape_[1]*shape_[2], shape_[3])
            pred_scale    =    pred_scale.view(shape_[0], shape_[1]*shape_[2], shape_[3])
            return pred_location, pred_scale, obs

        def compute_cosine_local_image_features(pred_location, pred_scale, obs):
            """ Returns cosine distance between predictions and observations
            """
            pred_location, pred_scale, obs = reshape_local_image_features(pred_location, pred_scale, obs)
            # compute cosine distance between ground truth image features and generated image features (zs)
            image_cosine = compute_cosine(pred_location, obs)
            image_cosine = torch.mean(image_cosine)
            return image_cosine

        def compute_cosine_global_image_features(pred_location, pred_scale, obs):
            """ Returns cosine distance between predictions and observations
            """
            pred_location, pred_scale, obs = reshape_global_image_features(pred_location, pred_scale, obs)
            # compute cosine distance between ground truth image features and generated image features (zs)
            image_cosine = compute_cosine(pred_location, obs)
            image_cosine = torch.mean(image_cosine)
            return image_cosine

        def compute_logprob_local_image_features(pred_location, pred_scale, obs):
            """ Returns log-likelihood of predictions given observation.
            """
            pred_location, pred_scale, obs = reshape_local_image_features(pred_location, pred_scale, obs)
            # pred_scale => [1, B, n_feats, image_features_dimensionality]
            pred_scale = torch.ones_like(pred_scale)
            # p_global_image_features => [1, B, n_feats, image_features_dimensionality]
            p_global_image_features = torch.distributions.Normal(pred_location, pred_scale)
            # logp_v => [1, B, n_feats, image_features_dimensionality]
            logp_v = p_global_image_features.log_prob( obs )
            # sum over image features dimensionality
            # log_pv => [1, B]
            logp_v = logp_v.sum(2).sum(1)
            # compute mean over mini-batch dimensionality
            # image_loss => [1]
            image_features_loss = logp_v.mean(0)
            #image_features_loss_data = image_features_loss.data.clone()
            return image_features_loss

        def compute_logprob_global_image_features(pred_location, pred_scale, obs):
            """ Returns log-likelihood of predictions given observation.
            """
            # pred_scale => [1, B, image_features_dimensionality]
            pred_scale = torch.ones_like(pred_scale)
            # p_global_image_features => [1, B, image_features_dimensionality]
            p_global_image_features = torch.distributions.Normal(pred_location, pred_scale)
            # logp_v => [1, B, image_features_dimensionality]
            logp_v = p_global_image_features.log_prob( obs )
            # sum over image features dimensionality
            # log_pv => [1, B]
            logp_v = logp_v.sum(1)
            # compute mean over mini-batch dimensionality
            # image_loss => [1]
            image_features_loss = logp_v.mean(1)
            return image_features_loss

        def compute_logprob_image_pixels(pred_location, pred_scale, obs):
            """ Returns log-likelihood of predictions given observation.
            """
            assert(pred_scale is None), 'Currently not supporting scales in image pixel prediction!'
            predicted_image = pred_location.squeeze(0)
            gtruth = obs.squeeze(0)
            gtruth = gtruth.type('torch.cuda.LongTensor')
            # compute negative log-likelihood between predicted and ground-truth image pixels
            image_pixels_loss = - self.image_loss_criterion(predicted_image, gtruth)
            return image_pixels_loss

        def compute_accuracy_image_pixels(pred_location, pred_scale, obs):
            assert(pred_scale is None), 'Currently not supporting scales in image pixel prediction!'
            # compute negative log-likelihood between predicted and ground-truth image pixels
            predicted_image = pred_location.squeeze(0)
            gtruth = obs.squeeze(0)
            gtruth = gtruth.type('torch.cuda.LongTensor')
            # compute image pixels' prediction accuracy
            _, predicted_idxs = predicted_image.max(1)
            correct_predictions = torch.eq(predicted_idxs, gtruth).double()
            image_pixels_acc = torch.mean(correct_predictions)
            return image_pixels_acc

        image_pixels_loss_data  = 0.
        image_pixels_acc_data   = 0.
        image_cosine_loss_data  = 0.
        image_logprob_loss_data = 0.

        # if there is some image loss to be computed, do it now
        if self.image_loss_type != 'none':
            if self.image_loss_type == 'categorical' or self.two_step_image_prediction:
                # predict image pixels - always use categorical
                image_pixels_loss = compute_logprob_image_pixels(
                        pred_location = p_image_pixels_location, 
                        pred_scale    = None, 
                        obs           = ground_truth_image_pixels)
                # compute accuracy of predicted image pixels
                image_pixels_acc = compute_accuracy_image_pixels(
                        pred_location = p_image_pixels_location, 
                        pred_scale    = None, 
                        obs           = ground_truth_image_pixels)
                image_pixels_loss_data   = image_pixels_loss.data.clone()
                image_pixels_acc_data    = image_pixels_acc.data.clone()

                # convert log-likelihood into negative log-likelihood
                image_pixels_loss = - image_pixels_loss

            if self.use_local_image_features and (self.image_loss_type != 'categorical' or self.two_step_image_prediction):
                #print("p_global_image_features_location.size(): ", p_global_image_features_location.size())
                #print("ground_truth_global_image_features.size(): ", ground_truth_global_image_features.size())
                # predict local image features
                image_cosine_loss = compute_cosine_local_image_features(
                        pred_location = p_global_image_features_location, 
                        pred_scale    = p_global_image_features_scale, 
                        obs           = ground_truth_global_image_features)
                image_logprob_loss = compute_logprob_local_image_features(
                        pred_location = p_global_image_features_location, 
                        pred_scale    = p_global_image_features_scale, 
                        obs           = ground_truth_global_image_features)
                image_cosine_loss_data   = image_cosine_loss.data.clone()
                image_logprob_loss_data  = image_logprob_loss.data.clone()

                # add the right loss to the ELBO
                # we will add this measure to the ELBO (i.e. will minimise it),
                if self.image_loss_type == 'cosine':
                    image_features_loss = - image_cosine_loss 
                elif self.image_loss_type == 'logprob':
                    image_features_loss = - image_logprob_loss
                else:
                    # defaults to 'logprob'
                    image_features_loss = - image_logprob_loss

            if not self.use_local_image_features and (self.image_loss_type != 'categorical' or self.two_step_image_prediction):
                # predict global image features
                image_cosine_loss = compute_cosine_global_image_features(
                        pred_location = p_global_image_features_location, 
                        pred_scale    = p_global_image_features_scale, 
                        obs           = ground_truth_global_image_features)
                image_logprob_loss = compute_logprob_global_image_features(
                        pred_location = p_global_image_features_location, 
                        pred_scale    = p_global_image_features_scale, 
                        obs           = ground_truth_global_image_features)
                #print("image_logprob_loss:", image_logprob_loss)
                image_cosine_loss_data   = image_cosine_loss.data.clone()
                image_logprob_loss_data  = image_logprob_loss.data.clone()

                # add the right loss to the ELBO
                # we will add this measure to the ELBO (i.e. will effectively minimise it), therefore the negative sign
                if self.image_loss_type == 'cosine':
                    image_features_loss = - image_cosine_loss
                elif self.image_loss_type == 'logprob':
                    image_features_loss = - image_logprob_loss
                else:
                    # defaults to 'logprob'
                    image_features_loss = - image_logprob_loss

            # the pytorch Variable `image_loss` below is what is added to the ELBO to be minimised
            # image_features_loss is already a negative log-likelihood
            if self.two_step_image_prediction:
                image_loss = image_features_loss + image_pixels_loss
            else:
                image_loss = image_features_loss if self.image_loss_type != 'categorical' else image_pixels_loss


        # 1. approximate a Dirichlet prior with a logistic normal
        def kl(params_i: list, params_j: list):
            """ 
            KL-divergence between two Normals: KL[N(u_i, s_i) || N(u_j, s_j)] 
            where params_i = [u_i, s_i] and similarly for j.

            Returns a tensor with the dimensionality of the location variable.
            """
            location_i, scale_i = params_i  # [mean, std]
            location_j, scale_j = params_j  # [mean, std]
            var_i = scale_i ** 2.
            var_j = scale_j ** 2.
            term1 = 1. / (2. * var_j) * ((location_i - location_j) ** 2. + var_i - var_j)
            term2 = torch.log(scale_j) - torch.log(scale_i)
            return term1 + term2 # tf.reduce_sum(term1 + term2, axis=-1)
        
        # latent variable KL-divergence (Z)
        # kl_stack => [1, B, z_latent_dim]
        kl_stack = kl([qz_location, qz_scale], [pz_location, pz_scale])
        # kl_stack => [1,B]
        kl_stack = torch.sum(kl_stack, dim=2) # sum over latent dimension
        # kl_stack => [1]
        kl_loss = torch.mean(kl_stack, dim=1) # mean over minibatch
        kl_loss_data_before = kl_loss.data.clone()

        if self.use_kl_annealing:
            # apply KL annealing (if not using KL annealing, kl_annealing_current will be 1.0)
            kl_loss *= self.kl_annealing_current
            
        if self.use_kl_freebits:
            kl_freebits_margin = torch.FloatTensor([self.kl_freebits_margin])
            if kl_loss.is_cuda:
                kl_freebits_margin = kl_freebits_margin.cuda()

            kl_freebits_margin = Variable( kl_freebits_margin, requires_grad=False)
            kl_loss = torch.max(kl_loss, kl_freebits_margin)
            #kl_loss = torch.max(kl_loss, torch.ones_like(kl_loss) * self.kl_freebits_margin)
            #print("kl loss after: %.2f"%float(kl_loss.data))
        kl_loss_data_after = kl_loss.data.clone()

        loss += (image_loss + kl_loss)
        if self.confidence < 1:
            elbo_loss_data = - likelihood.sum(0)
        else:
            elbo_loss_data = loss.data.clone()

        loss_data = {
                "nmt" : nmt_loss_data[0],
                "two_step_image_prediction" : self.two_step_image_prediction,
                "image_loss_type" : self.image_loss_type,
                "img_pixels_acc" : image_pixels_acc_data,
                "img_feats_loss" : image_logprob_loss_data,
                "img_pixels_loss" : image_pixels_loss_data,
                "img_feats_cos" : image_cosine_loss_data,
                "td_kl_before" : kl_loss_data_before,
                "td_kl_after" : kl_loss_data_after,
                "td_kl_multiplier" : self.kl_annealing_current,
                "elbo" : elbo_loss_data
        }
 
        # compute statistics
        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        # make sure that if current KL annealing went over 1.0 in a previous increment, put it at 1.0
        if self.kl_annealing_current > 1.0:
            self.kl_annealing_current = 1.0

        # anneal the KL
        if self.kl_annealing_current < 1.0 and self.n_model_updates >= self.kl_annealing_warmup_steps :
            self.kl_annealing_current += self.kl_annealing_increment

        #print("self.kl_annealing_current: %s"%str(self.kl_annealing_current))

        self.n_model_updates += 1

        return loss, stats

    def _stats(self, loss_data, scores, target):
        """
        Args:
            loss_data (:list[obj]:`FloatTensor`): list of components used to compute the ELBO loss,
                                                  computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`VIStatistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.VIStatistics(self.multimodal_model_type, loss_data, non_padding.sum(), num_correct)


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

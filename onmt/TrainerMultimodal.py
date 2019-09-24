from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import pdb
import time
import sys, os
import math
import tempfile
import pickle
import shutil
import gc

import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules
from onmt.Utils import MODEL_TYPES

from onmt.Trainer import Statistics


class VIStatistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, multimodal_model_type, loss_data = None, n_words=0, n_correct=0):
        assert(multimodal_model_type in MODEL_TYPES), "Invalid model type `%s`. Must be one of: %s"%(
                str(multimodal_model_type),str(MODEL_TYPES))

        if not loss_data is None:
            assert(("td_kl" in loss_data) or ("z_kl" in loss_data) or ("td_kl_before" in loss_data and "td_kl_after" in loss_data)), \
                "loss_data dictionary must have one of `td_kl` or `z_kl`."

        # this will contain important metrics computed for each model update
        self.progress_state_train = []
        self.progress_state_valid = []

        # tested with vi-model1 only
        self.two_step_image_prediction = loss_data["two_step_image_prediction"] if not loss_data is None and "two_step_image_prediction" in loss_data else False
        self.image_loss_type = loss_data["image_loss_type"] if not loss_data is None and "image_loss_type" in loss_data else 'logprob'
        self.image_feats_loss = loss_data["img_feats_loss"] if not loss_data is None and "img_feats_loss" in loss_data else 0.
        self.image_pixels_loss = loss_data["img_pixels_loss"] if not loss_data is None and "img_pixels_loss" in loss_data else 0.
        self.image_pixels_acc = loss_data["img_pixels_acc"] if not loss_data is None and "img_pixels_acc" in loss_data else 0.
        self.image_feats_cos  = loss_data["img_feats_cos"] if not loss_data is None and "img_feats_cos" in loss_data else 0.
        # we will print the average accuracy of image pixel prediction
        self.img_pixels_acc = loss_data["img_pixels_acc"] if not loss_data is None and "img_pixels_acc" in loss_data else 0.

        # tested with vi-model1, vi-model2, vi-model3
        self.nmt_loss   = loss_data["nmt"] if not loss_data is None else 0.
        self.elbo_loss  = loss_data["elbo"] if not loss_data is None else 0.
        self.te_kl      = loss_data["te_kl"] if not loss_data is None and "te_kl" in loss_data else 0.
        if not loss_data is None and "td_kl" in loss_data:
            # model ?!
            self.td_kl = loss_data["td_kl"]
            self.td_kl_before = 0.
            self.td_kl_after = 0.
            raise Exception("Bug 1")
        elif not loss_data is None and "td_kl_before" in loss_data and "td_kl_after" in loss_data:
            # model 1
            self.td_kl        = 0.
            self.td_kl_before = loss_data["td_kl_before"]
            self.td_kl_after  = loss_data["td_kl_after"]
        elif not loss_data is None and "z_kl" in loss_data:
            # model 1
            self.td_kl      = loss_data["z_kl"]
            raise Exception("Bug 3")
        else:
            self.td_kl = 0.
            self.td_kl_before = 0.
            self.td_kl_after = 0.

        if not loss_data is None and "td_kl_multiplier" in loss_data:
            self.td_kl_multiplier = loss_data["td_kl_multiplier"]
        else:
            self.td_kl_multiplier = 1.0

        self.multimodal_model_type = multimodal_model_type
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.n_updates = 0
        self.start_time = time.time()

    def update(self, stat):
        nmt_loss   = stat.nmt_loss

        if stat.two_step_image_prediction:
            image_feats_loss = stat.image_feats_loss if not stat.image_feats_loss is None else 0.
            image_pixels_loss = stat.image_pixels_loss if not stat.image_pixels_loss is None else 0.
        else:
            try:
                image_feats_loss = stat.image_loss if not stat.image_loss is None and stat.image_loss_type != 'categorical' else 0.
                image_pixels_loss = stat.image_loss if not stat.image_loss is None and stat.image_loss_type == 'categorical' else 0.
            except:
                image_feats_loss = stat.image_feats_loss if not stat.image_feats_loss is None and stat.image_loss_type != 'categorical' else 0.
                image_pixels_loss = stat.image_pixels_loss if not stat.image_pixels_loss is None and stat.image_loss_type == 'categorical' else 0.

        try:
            # image_cos in stat
            image_feats_cos = stat.image_cos
        except:
            image_cos_ = False
            image_feats_cos = stat.image_feats_cos

        elbo_loss  = stat.elbo_loss
        if self.multimodal_model_type in MODEL_TYPES:
            td_kl        = stat.td_kl if not stat.td_kl is None else 0.
            td_kl_before = stat.td_kl_before if not stat.td_kl_before is None else 0.
            td_kl_after  = stat.td_kl_after  if not stat.td_kl_after  is None else 0.
            td_kl_multiplier  = stat.td_kl_multiplier if not stat.td_kl_multiplier is None else 1.0

        te_kl      = stat.te_kl if not stat.te_kl is None else 0.

        # we will print the average accuracy of image pixel prediction
        img_pixels_acc = stat.img_pixels_acc if not stat.img_pixels_acc is None else 0.

        self.nmt_loss += nmt_loss
        self.image_feats_loss += image_feats_loss
        self.image_pixels_loss += image_pixels_loss
        self.image_feats_cos += image_feats_cos
        self.td_kl        += td_kl
        self.td_kl_before += td_kl_before
        self.td_kl_after  += td_kl_after
        self.td_kl_multiplier = td_kl_multiplier
        self.elbo_loss += elbo_loss
        self.img_pixels_acc += img_pixels_acc
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_updates += 1

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.nmt_loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()

        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; "+
               "td-kl-before (avg.): %6.2f; td-kl-after (avg.): %6.2f; td-kl-multiplier: %2.2f;"+
               "img-feats-loss (avg.): %6.2f; img-feats-cos (avg.): %6.2f; elbo (avg.): %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.td_kl_before / self.n_updates,
               self.td_kl_after / self.n_updates,
               self.td_kl_multiplier,
               self.image_feats_loss / self.n_updates,
               self.image_feats_cos / self.n_updates,
               self.elbo_loss / self.n_updates,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))

        self.n_updates = 0
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_img-feats-loss", float(self.image_feats_loss))
        experiment.add_scalar_value(prefix + "_img-feats-cos", float(self.image_feats_cos))
        experiment.add_scalar_value(prefix + "_img-pixels-loss", float(self.image_pixels_loss))
        experiment.add_scalar_value(prefix + "_img-pixels-acc", float(self.img_pixels_acc))
        experiment.add_scalar_value(prefix + "_td-kl-before", float(self.td_kl_before))
        experiment.add_scalar_value(prefix + "_td-kl-after", float(self.td_kl_after))
        experiment.add_scalar_value(prefix + "_td-kl", float(self.td_kl))
        experiment.add_scalar_value(prefix + "_td-kl-multiplier", float(self.td_kl_multiplier))
        experiment.add_scalar_value(prefix + "_elbo", float(self.elbo_loss))
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def save_progress(self, lr, model_updates, epoch, split):
        assert(split in ['train', 'valid']), "Must save progress for train/valid splits."

        t = self.elapsed_time()
        progress = {
            "epoch": epoch,
            "model_updates": model_updates,
            "elapsed_time": t,
            "ppl": self.ppl(),
            "acc": self.accuracy(),
            "image_feats_loss": float(self.image_feats_loss) / self.n_updates,
            "image_feats_cos": float(self.image_feats_cos) / self.n_updates,
            "img_pixels_loss": float(self.image_pixels_loss) / self.n_updates,
            "img_pixels_acc": float(self.img_pixels_acc) / self.n_updates,
            "td_kl_before": float(self.td_kl_before) / self.n_updates,
            "td_kl_after": float(self.td_kl_after) / self.n_updates,
            "td_kl": float(self.td_kl) / self.n_updates,
            "td_kl_multiplier": float(self.td_kl_multiplier),
            "elbo": float(self.elbo_loss) / self.n_updates,
            "tgt_per": self.n_words / t,
            "lr": lr
        }

        if split=='train':
            self.progress_state_train.append(progress)
        else: # split=='valid'
            self.progress_state_valid.append(progress)


class TrainerMultimodal(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            train_img_feats: training global image features.
            valid_img_feats: validation global image features.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 train_img_feats=None, valid_img_feats=None,
                 train_img_vecs=None, valid_img_vecs=None,
                 multimodal_model_type=None, model_updates=0,
                 model_opt=None, fields=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.train_img_feats = train_img_feats
        self.valid_img_feats = valid_img_feats
        self.train_img_vecs = train_img_vecs
        self.valid_img_vecs = valid_img_vecs
        self.multimodal_model_type = multimodal_model_type
        self.model_updates = model_updates

        # control early stopping
        self.model_opt = model_opt
        self.fields    = fields
        self.early_stop = onmt.EarlyStop.EarlyStop(model_opt.src, model_opt.tgt,
                model_opt.early_stopping_criteria,
                model_opt.start_early_stopping_at,
                model_opt.evaluate_every_n_model_updates,
                model_opt.patience,
                multimodal_model_type=multimodal_model_type,
                img_fname=model_opt.path_to_valid_img_feats,
                gpuid=model_opt.gpuid)

        self.n_model_updates = 0

        # hidden variable to keep internal representation of number of epochs trained
        self._epoch = 0

        assert(not (self.train_img_feats is None and self.train_img_vecs is None)), \
                'Must provide one of training image features/pixels!'
        assert(not (self.valid_img_feats is None and self.valid_img_vecs is None)), \
                'Must provide one of validation image features/pixels!'
        assert(self.multimodal_model_type in [None, 'vi-model1']), \
                'Invalid multimodal model type: %s!'%(self.multimodal_model_type)

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = VIStatistics(self.multimodal_model_type)
        report_stats = VIStatistics(self.multimodal_model_type)

        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                normalization += batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    if isinstance(total_stats, VIStatistics):
                        report_stats = report_func(
                                epoch, idx, num_batches,
                                total_stats.start_time, self.optim.lr,
                                report_stats, self.multimodal_model_type)
                    else:
                        report_stats = report_func(
                                epoch, idx, num_batches,
                                total_stats.start_time, self.optim.lr,
                                report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1
                self.n_model_updates += 1

            # early stopping?
            if not self.early_stop.early_stop_criteria in [None, 'perplexity']:
                # TODO: change overwrite=True to overwrite=self.model_opt.overwrite_model_file ?
                if self.n_model_updates % self.early_stop.evaluate_every_nupdates == 0:
                    temporary_model_fname, definitive_model_fname = \
                            self.drop_checkpoint(self.model_opt, epoch, self.fields,
                            valid_stats=None, overwrite=True,
                            checkpoint_type='best',
                            temporary=True)

                    is_current_model_best = self.early_stop.add_run(temporary_model_fname, self.n_model_updates)
                    if is_current_model_best:
                        self.drop_metric_scores(self.model_opt, epoch, self.fields,
                                valid_stats=None, overwrite=True,
                                checkpoint_type='best')
                        # overwrite file with best model with temporary model file
                        shutil.move(temporary_model_fname, definitive_model_fname)
                    else:
                        # delete temporary model file
                        os.unlink(temporary_model_fname)

                    # evaluate early stopping
                    if self.early_stop.signal_early_stopping:
                        break
            #else:
            #    # not early stopping, just save checkpoint overwriting any previous checkpoints
            #    self.drop_metric_scores(self.model_opt, epoch, self.fields,
            #            valid_stats=None, overwrite=True)

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)

            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        if not self.multimodal_model_type in MODEL_TYPES:
            stats = Statistics()
        else:
            stats = VIStatistics(self.multimodal_model_type)

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')
            # we are now interested in capturing the target sequences lengths
            if self.data_type == 'text':
                _, tgt_lengths = batch.tgt
                padding_token = self.train_loss.padding_idx
            else:
                tgt_lengths = None
            # set batch.tgt back to target tokens only (Loss object expects it like that)
            batch.tgt = batch.tgt[0]

            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy( self.valid_img_feats[idxs] )
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
            else:
                img_feats = img_feats.cpu()

            if self.model_opt.image_loss == 'categorical':
                # load image vectors for this minibatch into a pytorch Variable
                img_vecs = torch.from_numpy( self.valid_img_vecs[idxs] )
                img_vecs = torch.autograd.Variable(img_vecs, requires_grad=False)
                if next(self.model.parameters()).is_cuda:
                    img_vecs = img_vecs.cuda()
                else:
                    img_vecs = img_vecs.cpu()
            else:
                img_vecs = None

            # F-prop through the model.
            if self.multimodal_model_type in MODEL_TYPES:
                outputs, attns, _ = self.model(src, tgt, src_lengths, tgt_lengths, img_feats,
                        img_vecs=img_vecs, padding_token=padding_token)
            else:
                raise Exception("Multimodal model type not yet supported: %s"%(
                        self.multimodal_model_type))

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        if isinstance(batch_stats, VIStatistics):
            stats.save_progress(self.optim.lr, self.model_updates, self._epoch, 'valid')

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)


    def drop_metric_scores(self, opt, epoch, fields, valid_stats, overwrite=False, checkpoint_type='last'):
        """ Save metrics scores for a given checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
            overwrite (bool): whether to overwrite model file or save a new model file in each call to this function
            checkpoint_type (str): if 'last', save most current checkpoint (normal checkpointing). If 'best', save best model (used for model selection).
        """
        ct_options = ['last', 'best']
        assert(checkpoint_type in ct_options), \
                '`checkpoint_type` must take a value in %s . Received: %s'%(str(cp_options), checkpoint_type)

        checkpoint = {}

        #if not overwrite:
        #    model_fname = '%s_acc_%.2f_ppl_%.2f_e%d.pt'%(
        #                  opt.save_model, valid_stats.accuracy(), valid_stats.ppl(), epoch)
        #else:
        if checkpoint_type == 'best':
            if self.early_stop.early_stop_criteria == 'bleu':
                metric = 'bleu'
                # include only BLEU and METEOR scores for best model on validation set
                sorted_bleus = sorted(self.early_stop.results_bleu.items(), key=lambda kv:(kv[1], kv[0]))
                # best bleu and corresponding number of model updates (key)
                best_bleu = float([v for k,v in sorted_bleus][-1])
                n_updates_best_model = [k for k,v in sorted_bleus][-1]
                best_meteor = float(self.early_stop.results_meteor[ n_updates_best_model ])

            elif self.early_stop.early_stop_criteria == 'meteor':
                metric = 'meteor'
                # include only BLEU and METEOR scores for best model on validation set
                sorted_meteors = sorted(self.early_stop.results_meteor.items(), key=lambda kv:(kv[1], kv[0]))
                # best meteor and corresponding number of model updates (key)
                best_meteor = float([v for k,v in sorted_meteors][-1])
                n_updates_best_model = [k for k,v in sorted_meteors][-1]
                best_bleu = float(self.early_stop.results_bleu[ n_updates_best_model ])

            else:
                raise Exception('Metric not supported.')

            print("Trainer.drop_metric_scores - best BLEU: %.4f, best METEOR: %.4f"%(best_bleu, best_meteor))

            metrics_fname = '%s_BestModel%s.pkl'%(opt.save_model, metric.capitalize())
            checkpoint['n_updates'] = n_updates_best_model
            checkpoint['bleu'] = best_bleu
            checkpoint['meteor'] = best_meteor

        else:
            metrics_fname = '%s_MostCurrentModel.pkl'%(opt.save_model)

            # include all the previously computed BLEU and METEOR scores
            checkpoint['n_updates'] = self.n_model_updates
            checkpoint['bleu'] = list(self.early_stop.results_bleu.values())
            checkpoint['meteor'] = list(self.early_stop.results_meteor.values())

        #torch.save(checkpoint, model_fname)
        with open(metrics_fname, "wb") as f:
            pickle.dump(checkpoint, f, pickle.HIGHEST_PROTOCOL)


    def drop_checkpoint(self, opt, epoch, fields, valid_stats, train_stats=None, overwrite=False, checkpoint_type='last', temporary=False):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
            overwrite (bool): whether to overwrite model file or save a new model file in each call to this function
            checkpoint_type (str): if 'last', save most current checkpoint (normal checkpointing). If 'best', save best model (used for model selection).
        """
        ct_options = ['last', 'best']
        assert(checkpoint_type in ct_options), \
                '`checkpoint_type` must take a value in %s . Received: %s'%(str(cp_options), checkpoint_type)

        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }

        if temporary:
            # if we just want to save a temporary checkpoint and delete it afterwards
            tempf = tempfile.NamedTemporaryFile(delete=False)
            tempf_name = tempf.name
            tempf.close()

            temporary_model_fname = tempf_name

        if not overwrite:
            model_fname = '%s_acc_%.2f_ppl_%.2f_e%d.pt'%(
                          opt.save_model, valid_stats.accuracy(), valid_stats.ppl(), epoch)
        else:
            if checkpoint_type == 'best':
                if self.early_stop.early_stop_criteria == 'bleu':
                    metric = 'bleu'
                elif self.early_stop.early_stop_criteria == 'meteor':
                    metric = 'meteor'
                else:
                    raise Exception('Metric not supported.')

                model_fname = '%s_BestModel%s.pt'%(opt.save_model, metric.capitalize())
            else:
                model_fname = '%s_MostCurrentModel.pt'%(opt.save_model)

        if temporary:
            torch.save(checkpoint, temporary_model_fname)
            # return a tuple with the temporary model and its definitive name, in case it should be saved later
            returns = (temporary_model_fname, model_fname)
        else:
            torch.save(checkpoint, model_fname)
            # only return the (definitive) model file name
            returns = model_fname

        return returns


    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy( self.train_img_feats[idxs] )
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
            else:
                img_feats = img_feats.cpu()

            if self.model_opt.image_loss == 'categorical':
                # load image vectors for this minibatch into a pytorch Variable
                img_vecs = torch.from_numpy( self.train_img_vecs[idxs] )
                img_vecs = torch.autograd.Variable(img_vecs, requires_grad=False)
                if next(self.model.parameters()).is_cuda:
                    img_vecs = img_vecs.cuda()
                else:
                    img_vecs = img_vecs.cpu()
            else:
                img_vecs = None

            # work-around to get the maximum length for the minibatch
            target_size = torch.max(batch.tgt[1])
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')
            # we are now interested in capturing the target sequences lengths
            if self.data_type == 'text':
                _, tgt_outer_lengths = batch.tgt
                padding_token = self.train_loss.padding_idx
            else:
                tgt_lengths = None

            # set batch.tgt back to target tokens only (Loss object expects it like that)
            batch.tgt = batch.tgt[0]

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                tgt_lengths = tgt_outer_lengths[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                if self.multimodal_model_type in MODEL_TYPES:
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, tgt_outer_lengths, img_feats,
                                img_vecs=img_vecs, dec_state=dec_state, padding_token=padding_token)
                else:
                    raise Exception("Multimodal model type not yet supported: %s"%(
                            self.multimodal_model_type))

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 3.1. Update model updates counter
                self.model_updates += 1

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                #def save_progress(self, lr, model_updates, epoch, split):
                if isinstance(total_stats, VIStatistics):
                    total_stats.save_progress(self.optim.lr, self.model_updates, self._epoch, 'train')

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

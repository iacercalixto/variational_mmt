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
import time
import sys, os
import pdb
import math
import tempfile
import torch
import torch.nn as nn
import pickle
import shutil

import onmt
import onmt.io
import onmt.modules

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

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
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
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
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
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

        # control early stopping
        self.model_opt = model_opt
        self.fields    = fields
        self.early_stop = onmt.EarlyStop.EarlyStop(model_opt.src, model_opt.tgt,
                model_opt.early_stopping_criteria,
                model_opt.start_early_stopping_at,
                model_opt.evaluate_every_n_model_updates,
                model_opt.patience, multimodal_model_type=None, gpuid=model_opt.gpuid)
        self.n_model_updates = 0

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
        total_stats = Statistics()
        report_stats = Statistics()
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

        stats = Statistics()

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

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

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
            batch.tgt = batch.tgt[0]

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

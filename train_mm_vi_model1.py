#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random
import pdb

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.Utils
from onmt.Utils import use_gpu
import opts
import tables
import numpy

parser = argparse.ArgumentParser(
    description='train_mm_vi_model1.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
# variational multi-modal NMT parameters
opts.train_mm_vi_model1_opts(parser)
opt_ = parser.parse_args()

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if opt.early_stopping_criteria != 'perplexity':
    assert(opt.src and opt.tgt), 'Must provide path to validation src/tgt when not using perplexity.'

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    print("Using GPU")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)
    opt.cuda = opt.gpuid[0] > -1
else:
    print("Using CPU")
    torch.set_default_tensor_type("torch.FloatTensor")
 
if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# variational multimodal-NMT specific parameters
assert( os.path.isfile(opt.path_to_train_img_feats) ), \
        'Must provide the file containing the training image features.'

assert( os.path.isfile(opt.path_to_valid_img_feats) ), \
        'Must provide the file containing the validation image features.'

if opt.two_step_image_prediction:
    assert(opt.use_local_image_features), 'Can perform two-step image prediction '+\
            '(predict image features and pixels) only when using local image features.'


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats, multimodal_model_type):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(VIStatistics): old VIStatistics instance.
    Returns:
        report_stats(VIStatistics): updated VIStatistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.VIStatistics(multimodal_model_type)

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device,
            sort=False,
            train=self.is_train, sort_within_batch=True, repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            return sofar + max(len(new.tgt), len(new.src)) + 1

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, training=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    assert(not opt.copy_attn), 'ERROR: Not implemented.'

    if training:
        compute = onmt.VILoss.NMTVIModel1LossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing,
            use_kl_annealing=opt.use_kl_annealing,
            use_kl_freebits=opt.use_kl_freebits,
            kl_freebits_margin=opt.kl_freebits_margin,
            kl_annealing_current=opt.kl_annealing_start,
            kl_annealing_increment=opt.kl_annealing_increment,
            kl_annealing_warmup_steps=opt.kl_annealing_warmup_steps,
            image_loss_type=opt.image_loss,
            use_local_image_features=opt.use_local_image_features,
            two_step_image_prediction=opt.two_step_image_prediction,
        )
    else:
        # if testing/validating, use the whole KL as if there is no KL-annealing
        compute = onmt.VILoss.NMTVIModel1LossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing,
            use_kl_annealing=False,
            kl_annealing_current=opt.kl_annealing_start,
            kl_annealing_increment=opt.kl_annealing_increment,
            kl_annealing_warmup_steps=opt.kl_annealing_warmup_steps,
            image_loss_type=opt.image_loss,
            use_local_image_features=opt.use_local_image_features,
            two_step_image_prediction=opt.two_step_image_prediction,
        )

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, fields, optim, data_type,
                train_img_feats, valid_img_feats,
                train_img_vecs,  valid_img_vecs,
                model_opt):
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt, training=True)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt, training=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    multimodal_model_type = opt.multimodal_model_type

    trainer = onmt.TrainerMultimodal(model,
                           train_loss, valid_loss,
                           optim, trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count,
                           train_img_feats, valid_img_feats,
                           multimodal_model_type=multimodal_model_type,
                           train_img_vecs=train_img_vecs,
                           valid_img_vecs=valid_img_vecs,
                           model_opt=model_opt, fields=fields)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 0. Validate on the validation set.
        if epoch==1:
            valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                           fields, opt,
                                           is_train=False)
            valid_stats = trainer.validate(valid_iter)
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt, is_train=True)
        train_stats = trainer.train(train_iter, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())
        image_feats_loss = valid_stats.image_feats_loss
        image_feats_cos = valid_stats.image_feats_cos
        image_pixels_loss = valid_stats.image_pixels_loss
        image_pixels_acc = valid_stats.image_pixels_acc
        print('Validation image feats nll (avg.): %g' % (image_feats_loss / valid_stats.n_updates))
        print('Validation image fests cosine (avg.): %g' % (image_feats_cos / valid_stats.n_updates))
        #print('Validation image pixels nll (avg.): %g' % (image_pixels_loss / valid_stats.n_updates))
        #print('Validation image pixels acc (avg.): %g' % (image_pixels_acc / valid_stats.n_updates))

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if trainer.early_stop.early_stop_criteria in ['perplexity', None]:
            # not early-stopping
            if epoch >= opt.start_checkpoint_at:
                trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, overwrite=opt.overwrite_model_file)
        else:
            # if we are using a non-default early-stopping criteria
            # save model to use for continuing training later on if needed be
            model_name = trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, overwrite=opt.overwrite_model_file, checkpoint_type='last')
            trainer.drop_metric_scores(model_opt, epoch, fields, valid_stats, overwrite=True, checkpoint_type='last')
            print("")

        if trainer.early_stop.signal_early_stopping:
            print("WARNING: Early stopping!")
            break



def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    # make variational multi-modal NMT model
    model = onmt.ModelConstructor.make_vi_model_mmt(model_opt, fields,
                                                    use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint, finetune=False):
    if opt.train_from and not finetune:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim


def main():
    # start with loading the image features
    # open hdf5 file with the image features
    train_file = tables.open_file(opt.path_to_train_img_feats, mode='r')
    valid_file = tables.open_file(opt.path_to_valid_img_feats, mode='r')

    if opt.use_local_image_features:
        # load only local image features
        train_img_feats = train_file.root.local_feats[:]
        valid_img_feats = valid_file.root.local_feats[:]
        print('Using local image features...')
    elif opt.use_global_image_features:
        # load only global image features
        train_img_feats = train_file.root.global_feats[:]
        valid_img_feats = valid_file.root.global_feats[:]
        print('Using global image features...')
    else: # opt.use_posterior_image_features
        # load only image (unnormalised) posterior class probabilities (i.e. logits)
        train_img_feats = train_file.root.logits[:]
        valid_img_feats = valid_file.root.logits[:]
        print('Using image posterior class probabilities...')

    # close hdf5 file handlers
    train_file.close()
    valid_file.close()

    if opt.use_standardised_image_features:
        assert(not opt.use_posterior_image_features), \
                'Must not flag "standardising image features" when using class posteriors!'
        assert(not opt.use_local_image_features), \
                'Must not flag "standardising image features" when using local features!'

        try:
            mean_train_file = tables.open_file(opt.path_to_mean_train_img_feats, mode='r')
            std_train_file  = tables.open_file(opt.path_to_std_train_img_feats, mode='r')
            train_mean = mean_train_file.root.global_feats_mean[:]
            train_std  = std_train_file.root.global_feats_stds[:]
            mean_train_file.close()
            std_train_file.close()
        except:
            raise ValueError('Problem loading training set\'s mean and variance from `opt.path_to_mean_train_img_feats` and `opt.path_to_std_train_img_feats`.')

        # standardise training and validation image features
        train_img_feats = (train_img_feats - train_mean[None,:]) / train_std[None,:]
        valid_img_feats = (valid_img_feats - train_mean[None,:]) / train_std[None,:]

    if opt.image_loss == 'categorical':
        # we are using a categorical distribution to model the image pixels
        # we need to have the image vectors.
        assert(not opt.path_to_train_img_vecs is None and not opt.path_to_valid_img_vecs is None), \
                'Must provide the path to image vectors (-path_to_train_img_vecs and -path_to_valid_img_vecs) '+\
                'when using \'categorical\' as the image loss!'
        try:
            train_file = tables.open_file(opt.path_to_train_img_vecs, mode='r')
            valid_file = tables.open_file(opt.path_to_valid_img_vecs, mode='r')
            train_img_vecs = train_file.root.array[:]
            valid_img_vecs = valid_file.root.array[:]
            train_file.close()
            valid_file.close()
        except:
            raise ValueError('Problem loading `opt.path_to_train_img_vecs` and `opt.path_to_valid_img_vecs`.')
        
        # check for problems of colour/grayscale images
        if opt.use_rgb_images:
            assert(len(train_img_vecs.shape)==3 and train_img_vecs.shape[1] == train_img_vecs.shape[2]), \
                    'Image vectors must contain RGB dimension (i.e. 3). Found: %s'%str(train_img_vecs.shape)
            assert(len(valid_img_vecs.shape)==3 and valid_img_vecs.shape[1] == valid_img_vecs.shape[2]), \
                    'Image vectors must contain RGB dimension (i.e. 3). Found: %s'%str(valid_img_vecs.shape)
        else:
            # if the tensor has 3 dimensions, they are [batch, size, size]. unsqueeze dimension for the channel.
            if len(train_img_vecs.shape) == 3:
                #print("'train_img_vecs' has 3 dimensions. Introducing channel dimension...")
                ##shapes = train_img_vecs.shape
                #train_img_vecs = numpy.expand_dims(train_img_vecs, 1)
                print("train_img_vecs.shape: %s"%str(train_img_vecs.shape))
            if len(valid_img_vecs.shape) == 3:
                #print("'valid_img_vecs' has 3 dimensions. Introducing channel dimension...")
                ##shapes = valid_img_vecs.shape
                #valid_img_vecs = numpy.expand_dims(valid_img_vecs, 1)
                print("valid_img_vecs.shape: %s"%str(valid_img_vecs.shape))
    else:
        train_img_vecs = None
        valid_img_vecs = None

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        if not opt.finetune:
            model_opt = checkpoint['opt']
        else:
            model_opt = opt

        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint, opt.finetune)

    # Do training.
    train_model(model, fields, optim, data_type,
                train_img_feats, valid_img_feats,
                train_img_vecs,  valid_img_vecs,
                model_opt)


if __name__ == "__main__":
    main()

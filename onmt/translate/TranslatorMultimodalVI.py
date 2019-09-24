import sys
import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io
from onmt.Utils import MODEL_TYPES

class TranslatorMultimodalVI(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0,
                 test_img_feats=None, multimodal_model_type=None):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.test_img_feats = test_img_feats
        self.multimodal_model_type = multimodal_model_type

        assert(not test_img_feats is None),\
                'Please provide file with test image features.'
        assert(not multimodal_model_type is None),\
                'Please provide the multimodal model type name.'

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data, sent_idx):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           sent_idx: the sentence idxs mapping to the image features

        Todo:
           Shouldn't need the original dataset.
        """

        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy( self.test_img_feats[sent_idx] )
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)
        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # use image features as-is
        img_proj = img_feats

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        if self.multimodal_model_type in MODEL_TYPES:
            # standard encoder
            enc_states, context = self.model.encoder(src, src_lengths)
            
            # if model was trained using additional features to compute variational approximation,
            # use the generative network for prediction. if not, use the inference network.
            if self.model.conditional:
                q0, h = self.model.gen_net_global(context.detach(), src_lengths)
            else:
                q0, h = self.model.inf_net_global(context.detach(), src_lengths)

            s0 = q0.mean()

            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                                            src, context, enc_states)

        else:
            raise Exception("Multi-modal model not implemented: %s"%self.multimodal_model_type)

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)

        img_proj = img_proj.clone()
        # image features are in (batch x len x feats),
        # but rvar() function expects (len x batch x feats)
        img_proj = rvar(img_proj.transpose(0,1).data)
        # return it back to (batch x len x feats)
        img_proj = img_proj.transpose(0,1)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)
        if self.multimodal_model_type in MODEL_TYPES:
            # s0 => [1, # of topics]
            s0 = s0.repeat(beam_size, 1, 1)
        else:
            pass

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            if self.multimodal_model_type in MODEL_TYPES:
                dec_out, dec_states, attn = self.model.decoder(
                    inp, context, dec_states,
                    z_sample=s0.squeeze(1),
                    image_features=None,
                    context_lengths=context_lengths)

            else:
                raise Exception("Multi-modal model type not implemented: %s"%(
                    self.multimodal_model_type))

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :context_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data, sent_idx):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy( self.test_img_feats[sent_idx] )
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)

        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # use image features directly
        img_proj = img_feats

        if self.multimodal_model_type in MODEL_TYPES:
            # standard encoder
            enc_states, context = self.model.encoder(src, src_lengths)

            # if model was trained using additional features to compute variational approximation,
            # use the generative network for prediction. if not, use the inference network.
            if self.model.conditional:
                q0, h = self.model.gen_net_global(context.detach(), src_lengths)
            else:
                q0, h = self.model.inf_net_global(context.detach(), src_lengths)

            s0 = q0.mean()

            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                                            src, context, enc_states)

        else:
            raise Exception("Multi-modal model not implemented: %s"%self.multimodal_model_type)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        if self.multimodal_model_type in MODEL_TYPES:
            dec_out, dec_states, attn = self.model.decoder(
                inp, context, dec_states,
                z_sample=s0.squeeze(1),
                context_lengths=context_lengths)

        else:
            raise Exception("Multi-modal odel type not implemented: %s"%(
                self.multimodal_model_type))

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.modules.NormalVariationalEncoder import GlobalInferenceNetwork, \
                                                  ImageGlobalInferenceNetwork
from onmt.Utils import aeq
import sys
import numpy
from onmt.Models import *


class StdRNNVIModel1Decoder(RNNVIDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, word_dropout=0.0,
                 embeddings=None,
                 latent_dim=None,
                 reuse_copy_attn=False):
        self.multimodal_model_type = 'vi-model1'
        # initialise parent class
        super(StdRNNVIModel1Decoder, self).__init__(rnn_type,
                 bidirectional_encoder, num_layers,
                 hidden_size, # the embeddings will be concatenated to the latent variable
                 attn_type,
                 coverage_attn, context_gate,
                 copy_attn, dropout, word_dropout,
                 embeddings,
                 latent_dim,
                 reuse_copy_attn)

    def _run_forward_pass(self, input, context, state, context_lengths=None, **kwargs):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        ################################
        # begin: validate VI parameters
        ################################
        param_keys = ["z_sample", "image_features"]
        assert(all([key in kwargs for key in param_keys])), \
                "Must provide the following parameters in kwargs: %s . Received: %s"%(
                        str(param_keys), str(kwargs.keys()))
        z_sample = kwargs["z_sample"]
        image_features = kwargs["image_features"]
        assert(image_features is None), "Model 'vi-model1' does not use image features in the decoder! You should be using 'vi-model1.1' instead?"
        ##############################
        # end: validate VI parameters
        ##############################

        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)
        # word dropout drops (i.e. zeroes out) entire words rather then dimension of their embeddings
        emb = self.word_dropout(emb, training=True)

        # concatenate latent variable to embeddings
        z_sample = z_sample.unsqueeze(0).repeat( emb.size()[0], 1, 1)
        emb = torch.cat([emb, z_sample], 2)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size + self.latent_dim, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size +self.latent_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size



class InputFeedRNNVIModel1Decoder(RNNVIDecoderBase):
    """
    This decoder is based on the stochastic decoder (Schulz et al., 2018 to appear).

    A single global latent variable is used to initialise the hidden states of the decoder,
    and also to generate global image features.

    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, word_dropout=0.0,
                 embeddings=None,
                 latent_dim=None,
                 reuse_copy_attn=False):
        self.multimodal_model_type = 'vi-model1'
        # initialise parent class
        super(InputFeedRNNVIModel1Decoder, self).__init__(rnn_type,
                 bidirectional_encoder, num_layers,
                 hidden_size, attn_type,
                 coverage_attn, context_gate,
                 copy_attn, dropout, word_dropout,
                 embeddings,
                 latent_dim,
                 reuse_copy_attn)

    def _run_forward_pass(self, input, context, state,
            context_lengths=None,
            **kwargs):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        #############################
        # begin: validate parameters
        #############################
        param_keys = ["z_sample"]
        assert(all([key in kwargs for key in param_keys])), \
                "Must provide the following parameters in kwargs: %s . Received: %s"%(
                        str(param_keys), str(kwargs.keys()))
        z_sample = kwargs["z_sample"]

        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        #############################
        # end: validate parameters
        #############################

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim
        # word dropout drops (i.e. zeroes out) entire words rather then dimension of their embeddings
        emb = self.word_dropout(emb, training=True)

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None
       
        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            # compute the latent variable for the current decoder timestep
            # based on the previous decoder hidden state and sample from it.
            if len(hidden)==2: # LSTM
                hidden_ = hidden[0]
            else: # GRU
                # (must still do the same, since the hidden state is a tuple with one element)
                hidden_ = hidden[0]

            # the first dimension contains decoder layers, we only use the last.
            hidden_ = hidden_[-1,:,:]

            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output, z_sample], 1)
            rnn_output, hidden = self.rnn(emb_t, hidden)

            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors and latent variables.
        """
        return self.embeddings.embedding_size + self.hidden_size + self.latent_dim

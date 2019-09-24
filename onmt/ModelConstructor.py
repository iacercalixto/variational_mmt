"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu, MODEL_TYPES

# additional imports for multi-modal NMT
from onmt.Models import ImageGlobalFeaturesProjector, \
                        ImageLocalFeaturesProjector, \
                        NMTVIModel

from onmt.VI_Model1 import StdRNNVIModel1Decoder
from onmt.VI_Model1 import InputFeedRNNVIModel1Decoder

# additional imports for variational inference multi-modal NMT
from onmt.modules.NormalVariationalEncoder import GlobalInferenceNetwork, \
                                                  GlobalFullInferenceNetwork, \
                                                  ImageGlobalInferenceNetwork
#                                                  ImageTopicInferenceNetwork, \
#                                                  EmbeddingInferenceNetwork

def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.decoder_type == "doubly-attentive-rnn" and not opt.input_feed:
        return StdRNNDecoderDoublyAttentive(opt.rnn_type,
                             opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)
    elif opt.decoder_type == "doubly-attentive-rnn" and opt.input_feed:
        return InputFeedRNNDecoderDoublyAttentive(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    if 'multimodal_model_type' in opt:
        if opt.multimodal_model_type in MODEL_TYPES:
            print( 'Building variational multi-modal model...' )
            model = make_vi_model_mmt(model_opt, fields,
                                      use_gpu(opt), checkpoint)
        else:
            print( 'Building multi-modal model...' )
            model = make_base_model_mmt(model_opt, fields,
                                        use_gpu(opt), checkpoint)
    else:
        print( 'Building text-only model...' )
        model = make_base_model(model_opt, fields,
                                use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_encoder_image_global_features(opt):
    """
    Global image features encoder dispatcher function(s).
    Args:
        opt: the option in current environment.
    """
    # TODO: feat_size and num_layers only tested with vgg and resnet networks.
    # Validate that these values work for other CNN architectures as well.
    # infer dimensionality of the global image features
    if 'use_posterior_image_features' in opt and opt.use_posterior_image_features:
        feat_size = 1000
    else:
        if 'vgg' in opt.path_to_train_img_feats.lower():
            feat_size = 4096
        else:
            feat_size = 2048
    opt.global_image_features_dim = feat_size

    if opt.multimodal_model_type == 'imgw':
        num_layers = 2
    elif opt.multimodal_model_type == 'imge':
        num_layers = opt.enc_layers
    elif opt.multimodal_model_type == 'imgd':
        num_layers = opt.dec_layers
    elif opt.multimodal_model_type == 'vi':
        num_layers = opt.dec_layers

    return ImageGlobalFeaturesProjector(num_layers, feat_size, opt.rnn_size,
            opt.dropout_imgs, opt.use_nonlinear_projection)

def make_encoder_image_local_features(opt):
    """
    Local image features encoder dispatcher function(s).
    Args:
        opt: the option in current environment.
    """
    # TODO: feat_size and num_layers only tested with vgg network.
    # Validate that these values work for other CNN architectures as well.
    try:
        use_nonlinear_projection = opt.use_nonlinear_projection
    except:
        use_nonlinear_projection = False

    if 'vgg' in opt.path_to_train_img_feats.lower():
        feat_size = 512
    else:
        feat_size = 2048
    opt.local_image_features_dim = feat_size

    num_layers = 1
    return ImageLocalFeaturesProjector(num_layers, feat_size, opt.rnn_size,
            opt.dropout_imgs, use_nonlinear_projection)


def make_vi_model_mmt(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the VI multimodal NMT model.

        - `vi-model1`:
          a model where there is one global latent variable Z used to predict the image features
          and to inform the decoder initialisation.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # infer dimensionality of global image features
    if model_opt.use_posterior_image_features:
        feat_size = 1000
    else:
        if 'vgg' in model_opt.path_to_train_img_feats.lower():
            feat_size = 4096
        else:
            feat_size = 2048
    model_opt.global_image_features_dim = feat_size

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)

        if model_opt.multimodal_model_type in MODEL_TYPES:
            encoder = make_encoder(model_opt, src_embeddings)
        else:
            raise Exception("Multi-modal model type not implemented: %s"%
                            model_opt.multimodal_model_type)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')

    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    # image features encoder
    if model_opt.multimodal_model_type in MODEL_TYPES:
        if model_opt.use_posterior_image_features:
            image_features_type = "posterior"
            image_features_projector = None
        elif "use_local_image_features" in vars(model_opt) and model_opt.use_local_image_features:
            image_features_type = "local"
            image_features_projector = make_encoder_image_local_features(model_opt)
        else:
            assert(model_opt.use_global_image_features), 'Image features type not recognised. Choose from global, posterior, local.'
            image_features_type = "global"
            image_features_projector = None

    if "use_local_image_features" in vars(model_opt) and model_opt.use_local_image_features:
        image_feats_dim = model_opt.local_image_features_dim
    else:
        image_feats_dim = model_opt.global_image_features_dim

    if model_opt.multimodal_model_type in MODEL_TYPES:
        word_dropout = model_opt.word_dropout

        decoder = StdRNNVIModel1Decoder(model_opt.rnn_type, model_opt.brnn,
                                        model_opt.dec_layers, model_opt.rnn_size,
                                        model_opt.global_attention,
                                        model_opt.coverage_attn,
                                        model_opt.context_gate,
                                        model_opt.copy_attn,
                                        model_opt.dropout,
                                        word_dropout,
                                        tgt_embeddings,
                                        model_opt.z_latent_dim, # additional dimensionality is z_latent_dim
                                        model_opt.reuse_copy_attn)
    else:
        raise Exception('Model %s not implemented!'%str(model_opt.multimodal_model_type))

    if model_opt.multimodal_model_type in MODEL_TYPES:
        # if we are using a conditional model, it means we will train the variational approximation q
        # using all observations (x, y, v) and a generative network to predict z from x only.
        if model_opt.conditional:
            if image_features_type == 'local':
                # the reason to use 4 times the RNN is because we concatenate mean src encoding, mean tgt encoding,
                # and the result of an attention between the source and image feats, and between the target and image feats
                input_dims = 4 * model_opt.rnn_size

            else:
                input_dims = 2 * model_opt.rnn_size + model_opt.global_image_features_dim

            # this inference network uses x_1^m, y_1^n, v
            inf_net_global = GlobalFullInferenceNetwork(
                    model_opt.z_latent_dim,
                    input_dims,
                    "normal",
                    image_features_type=image_features_type
            )

            # use x_1^m to predict z
            gen_net_global = GlobalInferenceNetwork(model_opt.z_latent_dim, model_opt.rnn_size, "normal")

            # create bidirectional LSTM encoder to encode target sentences
            encoder_tgt = RNNEncoder(model_opt.rnn_type, True, model_opt.enc_layers,
                                     model_opt.rnn_size, model_opt.dropout, tgt_embeddings)

            # flow hidden dimension
            flow_h_dim = input_dims
        else:
            # use x_1^m to predict z
            inf_net_global = GlobalInferenceNetwork(model_opt.z_latent_dim, model_opt.rnn_size, "normal")
            gen_net_global = None
            # there is no target-language encoder
            encoder_tgt = None
            # flow hidden dimension
            flow_h_dim = model_opt.rnn_size

        # create a separate source-language encoder for the inference network
        encoder_inference = None

        if model_opt.non_shared_inference_network:
            #encoder_inference = make_encoder(model_opt, src_embeddings)
            src_embeddings_inference = make_embeddings(model_opt, src_dict,
                                                       feature_dicts)
            encoder_inference = MeanEncoder(model_opt.enc_layers, src_embeddings_inference)

        if "two_step_image_prediction" in vars(model_opt) and model_opt.two_step_image_prediction:
            if model_opt.use_local_image_features:
                image_feats_dim = model_opt.local_image_features_dim
            else:
                image_feats_dim = model_opt.global_image_features_dim

            if model_opt.use_local_image_features:
                # TODO remove hard-coded parameters into `opts.py`
                n_channels = [500, 1000]
                layer_dims = [3, 5]
                image_size  = 7 # predicting feature activations (7x7), not pixels
                inf_net_image_features = ImageDeconvolutionLocalFeatures(
                        input_size          = model_opt.z_latent_dim, 
                )

                # predict image pixels using output of the image features prediction (inf_net_image_features)
                # TODO remove hard-coded parameters into `opts.py`
                n_channels = [image_feats_dim, image_feats_dim//4]
                layer_dims = [7, 50]
                image_size = 100
                input_size = [2048, 7, 7]

                inf_net_image_pixels = ImageDeconvolution(
                        input_size          = input_size, 
                        image_size          = image_size, 
                        n_channels          = n_channels,
                        n_classes           = 256,
                        apply_log_softmax   = True,
                        layer_dims         = layer_dims,
                )

            else:
                # using global or posterior image features
                inf_net_image_features = ImageGlobalInferenceNetwork(
                        model_opt.z_latent_dim, 
                        image_feats_dim, 
                        model_opt.rnn_size, 
                        False, 
                        "normal"
                )

                # predict image pixels
                # TODO remove hard-coded parameters into `opts.py`
                n_channels = 3 if model_opt.use_rgb_images else 1
                n_channels = [n_channels]*2
                layer_dims = [25, 50]
                image_size = 100
                inf_net_image_pixels = ImageDeconvolution(model_opt.z_latent_dim, image_size=image_size, n_channels=n_channels)

            # we are predicting both image features (with image_loss == 'logprob') and image pixels (with image_loss == 'categorical')
            inf_net_image = (inf_net_image_features, inf_net_image_pixels)

        else:
            # we are only predicting either image features (image_loss != 'categorical') or image pixels (image_loss == 'categorical')
            if model_opt.image_loss != 'categorical':
                print("Creating image inference network")
                if model_opt.use_global_image_features or model_opt.use_posterior_image_features:
                    inf_net_image = ImageGlobalInferenceNetwork(
                            model_opt.z_latent_dim, 
                            model_opt.global_image_features_dim, 
                            model_opt.rnn_size, False, "normal")

                elif model_opt.use_local_image_features:
                    # TODO remove hard-coded parameters into `opts.py`
                    n_channels = [500, 1000]
                    layer_dims = [3, 5]
                    image_size  = 7 # predicting feature activations (7x7), not pixels
                    inf_net_image = ImageDeconvolutionLocalFeatures(
                            input_size          = model_opt.z_latent_dim, 
                    )

                else:
                    raise Exception("Image features type not recognised.")

                print(inf_net_image)
            else:
                # TODO remove hard-coded parameters into `opts.py`
                n_channels = 3 if model_opt.use_rgb_images else 1
                n_channels = [n_channels]*2
                image_size = 100
                
                inf_net_image = ImageDeconvolution(model_opt.z_latent_dim, image_size=image_size, n_channels=n_channels)

    else:
        raise Exception('Model %s not implemented!'%str(model_opt.multimodal_model_type))

    # Make NMTModel(= encoder + decoder).
    model = NMTVIModel(encoder, decoder,
                       encoder_inference = encoder_inference,
                       inf_net_global    = inf_net_global,
                       gen_net_global    = gen_net_global,
                       inf_net_image     = inf_net_image,
                       multimodal_model_type = 'vi-model1',
                       image_loss_type       = model_opt.image_loss,
                       image_features_type   = image_features_type,
                       image_features_projector = image_features_projector,
                       two_step_image_prediction = model_opt.two_step_image_prediction if "two_step_image_prediction" in vars(model_opt) else False,
                       conditional = model_opt.conditional,
                       encoder_tgt = encoder_tgt)

    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model

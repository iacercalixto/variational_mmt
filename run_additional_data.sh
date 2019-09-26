# this script assumes there are 2 GPU cards available in this machine (at least)
# please edit the script accordingly in case there are less GPU cards

DATA_PATH="/path/to/data/multi30k"
MODEL_PATH="/path/to/variational-multimodal-nmt-model-snapshots"
MODEL_FILE_NAME="MMT_VI_Model_AdditionalTrainingData"


# multi30k validation set
VAL_SRC="${DATA_PATH}/val.lc.norm.tok.bpe-en-de-30000.en"
VAL_TGT="${DATA_PATH}/val.lc.norm.tok.bpe-en-de-30000.de"
VAL_IMGS="${DATA_PATH}/flickr30k_valid_resnet50_cnn_features.hdf5"

# multi30k training set
TRAIN_SRC="${DATA_PATH}/train.lc.norm.tok.bpe-en-de-30000.en"
TRAIN_TGT="${DATA_PATH}/train.lc.norm.tok.bpe-en-de-30000.de"
TRAIN_IMGS="${DATA_PATH}/flickr30k_train_resnet50_cnn_features.hdf5"

# multi30k training set
PRETRAIN_SRC="${DATA_PATH}/flickr30k_train_translated-5x-comparable-1x.lc.norm.tok.bpe-en-de-30000.not-numbered.shuffled.en"
PRETRAIN_TGT="${DATA_PATH}/flickr30k_train_translated-5x-comparable-1x.lc.norm.tok.bpe-en-de-30000.not-numbered.shuffled.de"
PRETRAIN_IMGS="${DATA_PATH}/flickr30k_train_translated-5x-comparable-1x_resnet50_cnn_features.shuffled.hdf5"

# multi30k test set (2016)
TEST_2016_SRC="${DATA_PATH}/test_2016_flickr.lc.norm.tok.bpe-en-de-30000.en"
TEST_2016_TGT="${DATA_PATH}/test_2016_flickr.lc.norm.tok.bpe-en-de-30000.de"
TEST_2016_IMGS="${DATA_PATH}/flickr30k_test_resnet50_cnn_features.hdf5"

# multi30k test set (2017)
TEST_2017_SRC="${DATA_PATH}/test_2017_flickr.lc.norm.tok.bpe-en-de-30000.en"
TEST_2017_TGT="${DATA_PATH}/test_2017_flickr.lc.norm.tok.bpe-en-de-30000.de"
TEST_2017_IMGS="${DATA_PATH}/flickr30k_test_2017_flickr_resnet50_cnn_features.hdf5"

# ambiguous MSCOCO test set (2017)
TEST_2017_MSCOCO_SRC="${DATA_PATH}/test_2017_mscoco.lc.norm.tok.bpe-en-de-30000.en"
TEST_2017_MSCOCO_TGT="${DATA_PATH}/test_2017_mscoco.lc.norm.tok.bpe-en-de-30000.de"
TEST_2017_MSCOCO_IMGS="${DATA_PATH}/flickr30k_test_2017_mscoco_resnet50_cnn_features.hdf5"

# number of pre-training epochs (1 epochs == 290K examples)
PRETRAIN_EPOCHS=3
# number of fine-tuning epochs (1 epochs == 29K translated Multi30k examples)
EPOCHS=33
# LSTM hidden state size
RNN_SIZE=500
# latent variable size for fixed-prior model
Z_FIXED_SIZE=50
# latent variable size for conditional prior model
Z_COND_SIZE=500

#PRETRAIN_EPOCHS=3
#EPOCHS=2
#RNN_SIZE=10
#Z_FIXED_SIZE=10
#Z_COND_SIZE=10


##########
# train
##########

# pretrain the model on the concatenation of the translated Multi30k (~29K src/tgt/img instances)
# and the back-translated comparable Multi30k (additional ~145K src/tgt/img instances)
PRETRAIN_DATASET=${DATA_PATH}/concat-multi30k-translational-5times-comparable-1time-shuffled

# fine-tune the pre-trained models on the translated Multi30k only
FINETUNE_DATASET=${DATA_PATH}/m30k



# pre-train one conditional prior and one fixed-prior model
# one model on gpu 0, another one on gpu 1 (both spawn validation set translations on gpu 1)
python train_mm_vi_model1.py \
    -gpuid 0 -epochs ${PRETRAIN_EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size ${RNN_SIZE} --z_latent_dim ${Z_FIXED_SIZE} \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${PRETRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${PRETRAIN_DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior \
    -overwrite_model_file 2>&1 >> ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior.pretrain.log &

python train_mm_vi_model1.py \
    -gpuid 1 -epochs ${PRETRAIN_EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size ${RNN_SIZE} --z_latent_dim ${Z_COND_SIZE} \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${PRETRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${PRETRAIN_DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior \
    -overwrite_model_file \
    --conditional 2>&1 >> ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior.pretrain.log &

wait;


PRETRAINED_MODEL_FIXED=${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_MostCurrentModel.pt
PRETRAINED_MODEL_COND=${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_MostCurrentModel.pt

FINETUNE_MODEL_FIXED=${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_MostCurrentModel_finetuned.pt
FINETUNE_MODEL_COND=${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_MostCurrentModel_finetuned.pt

# create a copy of the pre-trained models for finetuning,
# so that we can still use pretrained models without fine-tuning if needed
cp ${PRETRAINED_MODEL_FIXED} ${FINETUNE_MODEL_FIXED}
cp ${PRETRAINED_MODEL_COND} ${FINETUNE_MODEL_COND}


# fine-tune each model (conditional- and fixed-prior) on the translated Multi30k only.
# one model on gpu 0, another one on gpu 1 (both spawn validation set translations on gpu 1)
python train_mm_vi_model1.py \
    -gpuid 0 -epochs ${EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size ${RNN_SIZE} --z_latent_dim ${Z_FIXED_SIZE} \
    -early_stopping_criteria 'bleu' \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${TRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${FINETUNE_DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_finetune \
    -overwrite_model_file \
    -train_from ${FINETUNE_MODEL_FIXED} \
    -finetune \
    2>&1 > ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior.finetune.log &

python train_mm_vi_model1.py \
    -gpuid 1 -epochs ${EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size ${RNN_SIZE} --z_latent_dim ${Z_COND_SIZE} \
    -early_stopping_criteria 'bleu' \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${TRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${FINETUNE_DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_finetune \
    -overwrite_model_file \
    -train_from ${FINETUNE_MODEL_COND} \
    -finetune \
    --conditional 2>&1 > ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior.finetune.log &

wait;



#############
# translate
#############

# translate the validation set
SPLIT="validation"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${VAL_SRC} \
    -path_to_test_img_feats ${VAL_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${VAL_SRC} \
    -path_to_test_img_feats ${VAL_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

# translate the test set (2016)
SPLIT="test2016"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${TEST_2016_SRC} \
    -path_to_test_img_feats ${TEST_2016_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${TEST_2016_SRC} \
    -path_to_test_img_feats ${TEST_2016_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

# translate the test set (2017)
SPLIT="test2017"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${TEST_2017_SRC} \
    -path_to_test_img_feats ${TEST_2017_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${TEST_2017_SRC} \
    -path_to_test_img_feats ${TEST_2017_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

# translate the ambiguous MSCOCO test set (2017)
SPLIT="test2017_mscoco"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${TEST_2017_MSCOCO_SRC} \
    -path_to_test_img_feats ${TEST_2017_MSCOCO_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${TEST_2017_MSCOCO_SRC} \
    -path_to_test_img_feats ${TEST_2017_MSCOCO_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

echo -ne "Finished. Translations of valid/test 2016/test 2017 (Flickr and ambiguous MSCOCO) can be found in:\n${MODEL_PATH}/${MODEL_FILE_NAME}.{fixed,conditional}-prior_BestModelBleu.pt.{validation,test2016,test2017,test2017_mscoco}-translations\n"

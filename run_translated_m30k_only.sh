# this script assumes there are 2 GPU cards available in this machine (at least)
# please edit the script accordingly in case there are less GPU cards

DATA_PATH="/path/to/data/multi30k"
MODEL_PATH="/path/to/variational-multimodal-nmt-model-snapshots"
MODEL_FILE_NAME="MMT_VI_Model_TranslatedM30K"


# multi30k validation set
VAL_SRC="${DATA_PATH}/val.lc.norm.tok.bpe-en-de-30000.en"
VAL_TGT="${DATA_PATH}/val.lc.norm.tok.bpe-en-de-30000.de"
VAL_IMGS="${DATA_PATH}/flickr30k_valid_resnet50_cnn_features.hdf5"

# multi30k training set
TRAIN_SRC="${DATA_PATH}/train.lc.norm.tok.bpe-en-de-30000.en"
TRAIN_TGT="${DATA_PATH}/train.lc.norm.tok.bpe-en-de-30000.de"
TRAIN_IMGS="${DATA_PATH}/flickr30k_train_resnet50_cnn_features.hdf5"

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

#EPOCHS=30
EPOCHS=1

##########
# train
##########

# train the model on the translated Multi30k data set only (~29K src/tgt/img instances)
DATASET=${DATA_PATH}/m30k

# train one conditional prior and one fixed-prior model
# one model on gpu 0, another one on gpu 1 (both spawn validation set translations on gpu 1)
python train_mm_vi_model1.py \
    -gpuid 0 -epochs ${EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size 500 --z_latent_dim 500 \
    -early_stopping_criteria 'bleu' \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${TRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior \
    -overwrite_model_file 2>&1 ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior.log &

python train_mm_vi_model1.py \
    -gpuid 1 -epochs ${EPOCHS} -batch_size 40 -valid_batch_size 40 -optim 'adam' -learning_rate 0.002 -rnn_type LSTM \
    -rnn_size 500 --z_latent_dim 500 \
    -early_stopping_criteria 'bleu' \
    -src ${VAL_SRC} \
    -tgt ${VAL_TGT} \
    -path_to_train_img_feats ${TRAIN_IMGS} \
    -path_to_valid_img_feats ${VAL_IMGS} \
    -data ${DATASET} \
    --multimodal_model_type  vi-model1 --use_global_image_features -dropout 0.5 -dropout_imgs 0.5 \
    -save_model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior \
    -overwrite_model_file \
    --conditional 2>&1 ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior.log &

wait;

#############
# translate
#############

# translate the validation set
SPLIT="validation"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${VAL_SRC} \
    -path_to_test_img_feats ${VAL_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${VAL_SRC} \
    -path_to_test_img_feats ${VAL_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

# translate the test set (2016)
SPLIT="test2016"
python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${TEST_2016_SRC} \
    -path_to_test_img_feats ${TEST_2016_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${TEST_2016_SRC} \
    -path_to_test_img_feats ${TEST_2016_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

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
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt \
    -src ${TEST_2017_MSCOCO_SRC} \
    -path_to_test_img_feats ${TEST_2017_MSCOCO_IMGS} \
    -gpu 0 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.fixed-prior_BestModelBleu.pt.${SPLIT}-translations &

python translate_mm_vi.py \
    -model ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt \
    -src ${TEST_2017_MSCOCO_SRC} \
    -path_to_test_img_feats ${TEST_2017_MSCOCO_IMGS} \
    -gpu 1 \
    -output ${MODEL_PATH}/${MODEL_FILE_NAME}.conditional-prior_BestModelBleu.pt.${SPLIT}-translations &

wait;

echo -ne "Finished. Translations of valid/test 2016/test 2017 (Flickr and ambiguous MSCOCO) can be found in:\n${MODEL_PATH}/${MODEL_FILE_NAME}.{fixed,conditional}-prior_BestModelBleu.pt.{validation,test2016,test2017,test2017_mscoco}-translations\n"


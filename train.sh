#!/bin/sh

. /content/shflags-master/shflags


DEFINE_integer "num_heads" 1 "Flag for number of heads"
DEFINE_integer "num_layers" 1 "Flag for number of layers"
DEFINE_integer "emb_dim" 128 "Flag for embedding dimension"
DEFINE_integer "seq_length" 512 "Flag for sequence lengths"
DEFINE_integer "vocab_size" 2 "Flag for vocab size"
DEFINE_integer "head_size" 64 "Flag for size of single dense layer head"
DEFINE_boolean "pos_embedding" true "Flag for weather to use positional embeddngs"
DEFINE_string "agg_method" "TOKEN" "Flag for aggregation method to be used"
DEFINE_string "pos_embedding_type" "SIN_COS" "Flag for which positional embedding to use"
DEFINE_integer "problem_id" 1 "Flag for problem id to be used"
DEFINE_string "optimizer" "adam" "Flag for optimizer to be used"
DEFINE_string "loss_function" "mean_squared_error" "Flag for loss function to be used"
DEFINE_string "metric" "mean_squared_error" "Flag for performance metric to be used"
DEFINE_float "learning_rate" 0.001 "Flag for learning_rate to be used"
DEFINE_integer "epochs" 15 "Flag for epochs to be used"
DEFINE_string "training" "true" "Flag for weather to perform training"


# parse the command-line
FLAGS "$@" || exit 1
eval set -- "${FLAGS_ARGV}"

python3 train.py --num_heads="${FLAGS_num_heads}" --num_layers=${FLAGS_num_layers} --emb_dim=${FLAGS_emb_dim} --seq_length=${FLAGS_seq_length} \
                --vocab_size="${FLAGS_vocab_size}" --head_size="${FLAGS_head_size}" --pos_embedding="${FLAGS_pos_embedding}" \
                --agg_method="${FLAGS_agg_method}" --pos_embedding_type="${FLAGS_pos_embedding_type}" --problem_id="${FLAGS_problem_id}" \
                --optimizer="${FLAGS_optimizer}" --loss_function="${FLAGS_loss_function}" --metric="${FLAGS_metric}" \
                --learning_rate="${FLAGS_learning_rate}" --epochs="${FLAGS_epochs}" --training="${FLAGS_training}"

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import time
import argparse
import os

from data import (
    read_data,
    create_vocs,
    create_labels_voc,
    build_vocab_GCN,
    get_indexes,
    get_batch_sup,
)

from evaluation import evaluate
from itertools import chain

from models.srl import SRL
from models.custom_allennlp.conditional_random_field import ConditionalRandomField

from pytorch_transformers import RobertaTokenizer, RobertaModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Span GCN training")
    # paths
    parser.add_argument(
        "--outputdir", type=str, default="savedir/", help="Output directory"
    )
    parser.add_argument("--outputmodelname", type=str, default="model.pickle")

    # training
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="learning rate, default 1e-3"
    )
    parser.add_argument(
        "--gradient-clipping", type=int, default=1, help="gradient clipping, default on"
    )

    # model
    parser.add_argument(
        "--emb-dim", type=int, default=100, help="word embedding dimension"
    )

    parser.add_argument(
        "--use-syntax", type=int, default=0, help="default do not use syntax"
    )

    parser.add_argument(
        "--use-elmo", type=int, default=0, help="default do not use ELMO embeddings"
    )
    parser.add_argument(
        "--use-bert", type=int, default=0, help="default do not use BERT embeddings"
    )

    parser.add_argument(
        "--embedding-layer-norm",
        type=int,
        default=0,
        help="default do not embedding layer norm",
    )

    parser.add_argument(
        "--enc-lstm-dim", type=int, default=300, help="encoder nhid dimension"
    )
    parser.add_argument("--n-layers", type=int, default=1, help="encoder num layers")
    parser.add_argument(
        "--n-layers-top", type=int, default=1, help="lstm layers after gcn"
    )

    parser.add_argument(
        "--word-drop", type=float, default=0.0, help="word dropout default 0.0, no drop"
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="2005",
        help="select the corpus, conll2005 is default conll2012 is the other option,",
    )

    parser.add_argument(
        "--train-file", type=str, required=True, help="path of the training file"
    )

    parser.add_argument(
        "--dev-file", type=str, required=True, help="path of the dev file"
    )

    parser.add_argument(
        "--glove-path",
        type=str,
        required=True,
        help="path of Glove glove.6B.100d.txt embeddings",
    )

    parser.add_argument(
        "--bilinear-dropout",
        type=float,
        default=0.0,
        help="dropout at the bilinear module",
    )
    parser.add_argument(
        "--gcn-dropout", type=float, default=0.0, help="dropout of the gcn input module"
    )
    parser.add_argument(
        "--emb-dropout",
        type=float,
        default=0.0,
        help="dropout of the embedding , default off",
    )
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=0.0,
        help="dropout of the gcn edges , default off",
    )
    parser.add_argument(
        "--non-linearity",
        type=str,
        default="relu",
        help="nonlinearity used, default relu",
    )

    # gpu
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    params, _ = parser.parse_known_args()

    # set gpu device
    torch.cuda.set_device(params.gpu_id)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    GLOVE_PATH = params.glove_path

    train_file = params.train_file
    dev_file = params.dev_file

    train, train_data_file, w_c_to_idx, c_c_to_idx = read_data(train_file, {}, {})
    print("train examples", len(train))
    dev, dev_data_file, w_c_to_idx, c_c_to_idx = read_data(
        dev_file, w_c_to_idx, c_c_to_idx
    )
    print("dev examples", len(dev))

    word_to_idx, pos_to_idx = create_vocs(train)
    roles_to_idx, idx_to_roles = create_labels_voc(train + dev)
    word_vec = build_vocab_GCN(
        [t["text"] for t in train] + [t["text"] for t in dev], GLOVE_PATH
    )
    train = get_indexes(train, word_to_idx, pos_to_idx, roles_to_idx)
    dev = get_indexes(dev, word_to_idx, pos_to_idx, roles_to_idx)

    srl = SRL(
        params.enc_lstm_dim,
        len(roles_to_idx),
        params.n_layers,
        len(w_c_to_idx),
        len(c_c_to_idx),
        params.use_syntax,
        params.embedding_layer_norm,
        params.n_layers_top,
        params.use_elmo,
        params.use_bert,
        params,
        params.gpu_id,
    )

    print(srl)

    crf = ConditionalRandomField(
        len(roles_to_idx), None, include_start_end_transitions=True
    )
    print(crf)

    model_parameters = filter(
        lambda p: p.requires_grad, chain(srl.parameters(), crf.parameters())
    )

    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total parameters =", num_params)
    print(params)

    if params.use_bert:
        bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        bert_model = RobertaModel.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        if params.gpu_id > -1:
            bert_model.cuda()
    else:
        bert_tokenizer = None
        bert_model = None
    if params.gpu_id > -1:
        srl.cuda()
        crf.cuda()

    lr = params.learning_rate
    # optimizer

    optimizer = optim.Adam(
        chain(srl.parameters(), crf.parameters()),
        lr=lr,
        weight_decay=params.weight_decay,
    )

    first_optim = True

    val_acc_best = -1.0  # set the best validation accuracy to -1.0
    adam_stop = False
    stop_training = 20

    for epc in range(params.n_epochs):
        srl.train()
        all_costs = []
        logs = []
        np.random.shuffle(train)
        last_time = time.time()
        train_len = len(train)
        for stidx in range(0, train_len, params.batch_size):

            labels_batch, sentences_batch, predicate_flags_batch, mask_batch, lengths_batch, fixed_embs, constituents, GCN_w_c, GCN_c_w, GCN_c_c, mask_const_batch, predicate_index, elmo_character_ids, bert_embs = get_batch_sup(
                train[stidx : stidx + params.batch_size],
                word_vec,
                params.gpu_id,
                params.enc_lstm_dim,
                params.word_drop,
                bert_tokenizer,
                bert_model,
            )

            output = srl(
                sentences_batch,
                predicate_flags_batch,
                mask_batch,
                lengths_batch,
                fixed_embs,
                constituents,
                GCN_w_c,
                GCN_c_w,
                GCN_c_c,
                mask_const_batch,
                predicate_index,
                elmo_character_ids,
                bert_embs,
            )

            # CRF Log-likelihood loss
            log_likelihood = crf(output, labels_batch, mask_batch)
            all_costs.append(-log_likelihood.item())
            # backward

            optimizer.zero_grad()

            total_loss = -log_likelihood

            # optimizer step
            total_loss.backward()

            # Gradient clipping
            if params.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    chain(srl.parameters(), crf.parameters()), 1
                )

            optimizer.step()

            if params.corpus == "2005":
                stop_every = 10000 // params.batch_size
            else:
                stop_every = 30000 // params.batch_size

            start_check_at = 2
            if (
                len(all_costs) == stop_every
            ):  # print training statistics every 100 batches
                print(
                    "Training for " + str(stop_every) + " batches took",
                    str(round((time.time() - last_time) / 60, 2)),
                    "minutes.",
                )

                logs.append(
                    "{0} ; loss {1}".format(stidx, round(np.mean(all_costs), 4))
                )
                if epc > start_check_at:

                    eval_acc, val_acc_best, stop_training, adam_stop = evaluate(
                        srl,
                        epc + 1,
                        dev,
                        val_acc_best,
                        word_vec,
                        idx_to_roles,
                        params,
                        params.outputmodelname,
                        params.outputdir,
                        crf,
                        adam_stop,
                        stop_training,
                        bert_tokenizer,
                        bert_model,
                        eval_type="valid",
                        final_eval=False,
                        gold_data_file=dev_data_file,
                        gold_file_path=dev_file,
                    )

                srl.train()
                crf.train()
                print(logs[-1])
                all_costs = []
                last_time = time.time()

        print("epoch " + str(epc + 1) + " done.")

        eval_acc, val_acc_best, stop_training, adam_stop = evaluate(
            srl,
            epc + 1,
            dev,
            val_acc_best,
            word_vec,
            idx_to_roles,
            params,
            params.outputmodelname,
            params.outputdir,
            crf,
            adam_stop,
            stop_training,
            bert_tokenizer,
            bert_model,
            eval_type="valid",
            final_eval=False,
            gold_data_file=dev_data_file,
            gold_file_path=dev_file,
        )

        srl.train()
        crf.train()

        if stop_training:
            adam_stop = 20
            stop_training = False
            lr = lr * 0.5
            print("Learning rate reduced to", str(lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if lr < 0.000125:
                break

    srl.load_state_dict(
        torch.load(os.path.join(params.outputdir, params.outputmodelname))
    )

    crf.load_state_dict(
        torch.load(os.path.join(params.outputdir, params.outputmodelname + "crf"))
    )
    evaluate(
        srl,
        1000,
        dev,
        val_acc_best,
        word_vec,
        idx_to_roles,
        params,
        params.outputmodelname,
        params.outputdir,
        crf,
        adam_stop,
        stop_training,
        bert_tokenizer,
        bert_model,
        eval_type="valid",
        final_eval=False,
        gold_data_file=dev_data_file,
        gold_file_path=dev_file,
    )

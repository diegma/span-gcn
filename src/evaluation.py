import torch
import os
from subprocess import check_output

from data import get_batch_sup, get_batch_sup_frames
import copy
import time
import numpy as np
import os.path


def evaluate(
    model,
    epoch,
    data,
    val_acc_best,
    word_vec,
    idx_to_roles,
    params,
    modelname,
    save_dir,
    crf,
    adam_stop,
    stop_training,
    bert_tokenizer,
    bert_model,
    eval_type="valid",
    final_eval=False,
    gold_data_file=None,
    gold_file_path=None,
):
    last_time = time.time()

    srl = model
    srl.eval()
    crf.eval()

    if eval_type == "valid":
        print("\nVALIDATION : Epoch {0}".format(epoch))

    batch_size = 16

    all_pred = []
    data_len = len(data)
    for stidx in range(0, data_len, batch_size):
        labels_batch, sentences_batch, predicate_flags_batch, mask_batch, lengths_batch, fixed_embs, constituents, GCN_w_c, GCN_c_w, GCN_c_c, mask_const_batch, predicate_index, elmo_character_ids, bert_embs = get_batch_sup(
            data[stidx : stidx + batch_size],
            word_vec,
            params.gpu_id,
            params.enc_lstm_dim,
            0.0,
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
        best_paths = crf.viterbi_tags(output, mask_batch)

        for x, _ in best_paths:
            all_pred += x

    # if params.corpus == "2005_elmo" or params.corpus == "2005":
    #     corp = "2005"
    #
    # elif params.corpus == "2012_elmo" or params.corpus == "2012":
    #     corp = "2012"
    # else:
    #     corp = params.corpus

    print("Eval took", str(round((time.time() - last_time) / 60, 2)))
    if gold_data_file:
        try:
            annotated_data = _prep_conll_predictions(
                all_pred, gold_data_file, idx_to_roles
            )
            _print_conll_predictions(annotated_data, modelname + "_" + ".txt")

            # if final_eval:
            # if eval_type == "valid":
            # gold_standard_file = "data/conll" + corp + "/evaluation_files/dev-set"
            gold_standard_file = gold_file_path
            precision, recall, eval_acc = _evaluate_conll(
                modelname + "_" + ".txt", gold_standard_file
            )

            # elif eval_type == "test":
            #     gold_standard_file = "data/conll" + corp + "/evaluation_files/test-set"
            #     precision, recall, eval_acc = _evaluate_conll(
            #         params.outputmodelname + "_" + eval_type + ".txt",
            #         gold_standard_file,
            #     )
            #
            # elif eval_type == "ood":
            #     gold_standard_file = "data/conll" + corp + "/evaluation_files/ood-set"
            #     precision, recall, eval_acc = _evaluate_conll(
            #         params.outputmodelname + "_" + eval_type + ".txt",
            #         gold_standard_file,
            #     )

        except IndexError:
            print(all_pred),
            print(gold_data_file)
            print(idx_to_roles)
            # precision, recall, eval_acc = 0.0, 0.0, 0.0

    if final_eval:
        print(
            "finalgrep : F1 {0} : {1} precision {0} : {2} recall {0} : {3}".format(
                eval_type, eval_acc, precision, recall
            )
        )
    else:
        print(
            "togrep : results : epoch {0} ; F1 score {1} : {2} "
            "precision {1} : {3} recall {1} : {4}".format(
                epoch, eval_type, eval_acc, precision, recall
            )
        )

    if eval_type == "valid" and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print("saving model at epoch {0}".format(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, modelname))
            torch.save(crf.state_dict(), os.path.join(save_dir, modelname + "crf"))
            val_acc_best = eval_acc
            adam_stop = 20
            stop_training = False
        else:
            # early stopping
            # stop_training = adam_stop
            adam_stop -= 1
            if adam_stop == 0:
                stop_training = True
    return eval_acc, val_acc_best, stop_training, adam_stop


def evaluate_frames(
    model,
    epoch,
    data,
    val_acc_best,
    word_vec,
    roles_to_idx,
    roles_to_idx_all,
    idx_to_roles,
    idx_to_roles_all,
    params,
    crf,
    adam_stop,
    stop_training,
    pred_constraints,
    span_metric,
    frame_to_idx,
    eval_type="valid",
    final_eval=False,
    gold_data_file=None,
):
    last_time = time.time()

    srl = model
    srl.eval()
    crf.eval()

    if eval_type == "valid":
        print("\nVALIDATION : Epoch {0}".format(epoch))

    batch_size = 8

    data_len = len(data)
    for stidx in range(0, data_len, batch_size):
        labels_batch, sentences_batch, predicate_flags_batch, mask_batch, lengths_batch, fixed_embs, constituents, GCN_w_c, GCN_c_w, GCN_c_c, mask_const_batch, predicate_index, softmax_constraints, target_indexes, frames, frames_emb_batch = get_batch_sup_frames(
            data[stidx : stidx + batch_size],
            word_vec,
            params.gpu_id,
            roles_to_idx_all,
            params.enc_lstm_dim,
            0.0,
            pred_constraints,
            frame_to_idx,
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
            softmax_constraints,
            frames_emb_batch,
        )

        best_paths = crf.viterbi_tags(output, mask_batch)
        # Just get the tags and ignore the score.
        max_len = 0
        for bp, _ in best_paths:
            max_len = max(max_len, len(bp))
        pred = np.zeros((batch_size, max_len))

        for i, x in enumerate(best_paths):
            for id, y in enumerate(x[0]):
                pred[i, id] = y

        span_metric(
            pred, labels_batch, mask_batch, target_indices=target_indexes, frames=frames
        )

    metric_dict = span_metric.get_metric(reset=True)

    # This can be a lot of metrics, as there are 3 per class.
    # we only really care about the overall metrics, so we filter for them here.
    # print ([x,y for x, y in metric_dict.items() if "overall" in x])
    # print (metric_dict)
    # print(metric_dict['f1-measure-overall'], metric_dict['precision-overall'], metric_dict['recall-overall'])
    eval_acc = metric_dict["f1-measure-overall"]
    print("Eval took", str(round((time.time() - last_time) / 60, 2)))

    f1 = round(metric_dict["f1-measure-overall"] * 100, 2)
    prec = round(metric_dict["precision-overall"] * 100, 2)
    rec = round(metric_dict["recall-overall"] * 100, 2)

    if final_eval:
        print(
            "finalgrep : F1 {0} : {1} precision {0} : {2} recall {0} : {3}".format(
                eval_type, f1, prec, rec
            )
        )
    else:
        print(
            "togrep : results : epoch {0} ; F1 score {1} : {2} "
            "precision {1} : {3} recall {1} : {4}".format(
                epoch, eval_type, f1, prec, rec
            )
        )

    if eval_type == "valid" and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print("saving model at epoch {0}".format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(
                model.state_dict(),
                os.path.join(params.outputdir, params.outputmodelname),
            )
            torch.save(
                crf.state_dict(),
                os.path.join(params.outputdir, params.outputmodelname + "crf"),
            )
            val_acc_best = eval_acc
            adam_stop = 20
            stop_training = False
        else:
            # early stopping
            # stop_training = adam_stop
            adam_stop -= 1
            if adam_stop == 0:
                stop_training = True
    return eval_acc, val_acc_best, stop_training, adam_stop


def _prep_conll_predictions(pred, conll_gold, idx_to_roles):
    data = copy.deepcopy(conll_gold)
    cur_sent_len = 0
    sent_lenghts = []
    for li, line in enumerate(data):
        if len(line) == 0:
            sent_lenghts.append(cur_sent_len)
            cur_sent_len = 0
        else:
            cur_sent_len += 1
    curr_sent = 0
    line_count = 0
    n_predicates = 0
    prev_open = []
    for li, line in enumerate(data):
        if len(line) == 0:
            if n_predicates > 0:
                line_count += sent_lenghts[curr_sent] * (n_predicates - 1)
                for la, label in enumerate(prev_open):
                    if label != 0:
                        data[li - 1][la + 6] += ")"
            prev_open = []
            curr_sent += 1
        else:
            if len(prev_open) == 0:
                for _ in line[6:]:
                    prev_open.append(0)
            for la, label in enumerate(line[6:]):
                if pred[line_count + (la * sent_lenghts[curr_sent])] >= len(
                    idx_to_roles
                ):
                    lb = "O"
                else:
                    lb = idx_to_roles[pred[line_count + (la * sent_lenghts[curr_sent])]]
                if lb[0] == "O":
                    data[li][la + 6] = "*"
                    if prev_open[la]:
                        data[li - 1][la + 6] += ")"
                    prev_open[la] = 0
                elif lb[0] == "B":
                    if prev_open[la]:
                        data[li - 1][la + 6] += ")"
                    data[li][la + 6] = "(" + lb[2:] + "*"
                    prev_open[la] = lb[2:]
                elif lb[0] == "I":
                    if not prev_open[la]:
                        data[li][la + 6] = "(" + lb[2:] + "*"
                        prev_open[la] = lb[2:]
                    elif lb[2:] != prev_open[la]:
                        data[li - 1][la + 6] += ")"
                        data[li][la + 6] = "(" + lb[2:] + "*"
                        prev_open[la] = lb[2:]
                    else:
                        data[li][la + 6] = "*"
            n_predicates = len(line[6:])

            if n_predicates > 0:
                line_count += 1
    return data


def _print_conll_predictions(data, name):
    with open("data/predictions/" + name, "w") as out:
        for line in data:
            out.write(" ".join(line[5:]) + "\n")


def _evaluate_conll(prediction_file, gold_standard_file):
    script = "data/scripts/srl-eval.pl"

    cut_script_args = ["cut", "-d", " ", "-f", "10-100", gold_standard_file]

    eval_script_args = [
        script,
        "/tmp/" + gold_standard_file.split("/")[-1],
        "data/predictions/" + prediction_file,
    ]

    try:
        DEVNULL = open(os.devnull, "wb")
        cut_out = check_output(cut_script_args, stderr=DEVNULL)
        open("/tmp/" + gold_standard_file.split("/")[-1], "wb").write(cut_out)

        out = check_output(eval_script_args, stderr=DEVNULL)
        out = out.decode("utf-8")

        out_ = " ".join(out.split())
        all_ = out_.split()

        open("data/predictions/" + prediction_file + "_eval.out", "w").write(out)
        prec = all_[27]
        rec = all_[28]
        f1 = all_[29]
        return float(prec), float(rec), float(f1)
    except:
        raise IOError

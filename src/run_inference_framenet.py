import torch
import numpy as np
import argparse
import os

from data import (
    read_data_frames,
    create_vocs,
    create_labels_voc,
    build_vocab_GCN,
    get_indexes,
    create_constraints,
    create_predicate_constraints,
)

from evaluation import evaluate_frames
from itertools import chain
from span_eval.NonBioSpanBasedF1Measure import NonBioSpanBasedF1Measure

from models.srl import SRL_Framenet
from models.custom_allennlp.conditional_random_field import ConditionalRandomField

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Span GCN Inference framenet")
    # paths
    parser.add_argument("--dir", type=str, default="savedir/", help="Output directory")
    parser.add_argument("--modelname", type=str, default="model.pickle")

    parser.add_argument("--batch-size", type=int, default=8)

    # model
    parser.add_argument(
        "--train-file", type=str, required=True, help="path of the training file"
    )

    parser.add_argument(
        "--dev-file", type=str, required=True, help="path of the dev file"
    )

    parser.add_argument(
        "--test-file", type=str, required=True, help="path of the inference file"
    )

    parser.add_argument(
        "--glove-path",
        type=str,
        required=True,
        help="path of Glove glove.6B.100d.txt embeddings",
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        required=True,
        help="path of framenet ontologies, e.g., fndata-1.5",
    )

    parser.add_argument(
        "--emb-dim", type=int, default=100, help="word embedding dimension"
    )
    parser.add_argument(
        "--use-syntax", type=int, default=0, help="default do not use syntax"
    )

    parser.add_argument(
        "--embedding-layer-norm",
        type=int,
        default=0,
        help="default do not embedding layer norm",
    )

    parser.add_argument(
        "--enc-lstm-dim", type=int, default=512, help="encoder nhid dimension"
    )
    parser.add_argument("--n-layers", type=int, default=1, help="encoder num layers")
    parser.add_argument(
        "--n-layers-top", type=int, default=1, help="lstm layers after gcn"
    )
    parser.add_argument(
        "--word-drop", type=float, default=0.0, help="word dropout default 0.0, no drop"
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
        type=int,
        default=0,
        help="dropout of the embedding , default off",
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
    test_file = params.test_file

    train, train_data_file, w_c_to_idx, c_c_to_idx, frame_to_idx = read_data_frames(
        train_file, {}, {}, True
    )
    dev, dev_data_file, w_c_to_idx, c_c_to_idx, _ = read_data_frames(
        dev_file, w_c_to_idx, c_c_to_idx, False
    )

    test, test_data_file, _, _, _ = read_data_frames(
        test_file, w_c_to_idx, c_c_to_idx, False
    )

    word_to_idx, pos_to_idx = create_vocs(train)
    roles_to_idx, idx_to_roles = create_labels_voc(train + dev)
    word_vec = build_vocab_GCN(
        [t["text"] for t in train]
        + [t["text"] for t in dev]
        + [t["text"] for t in test],
        GLOVE_PATH,
    )

    test = get_indexes(test, word_to_idx, pos_to_idx, roles_to_idx)

    srl = SRL_Framenet(
        params.enc_lstm_dim,
        len(roles_to_idx),
        params.n_layers,
        len(w_c_to_idx),
        len(c_c_to_idx),
        params.use_syntax,
        params.embedding_layer_norm,
        params.n_layers_top,
        params,
        params.gpu_id,
    )

    print(srl)

    constraints = create_constraints(train, roles_to_idx)
    print(len(constraints))

    pred_constraints = create_predicate_constraints(train, roles_to_idx)
    crf = ConditionalRandomField(
        len(roles_to_idx), constraints, include_start_end_transitions=True
    )
    print(crf)
    model_parameters = filter(
        lambda p: p.requires_grad, chain(srl.parameters(), crf.parameters())
    )

    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total parameters =", num_params)
    print(params)

    if params.gpu_id > -1:
        srl.cuda()
        crf.cuda()

    srl.load_state_dict(torch.load(os.path.join(params.dir, params.modelname)))
    crf.load_state_dict(torch.load(os.path.join(params.dir, params.modelname + "crf")))

    first_optim = True
    val_acc_best = -1.0  # set the best validation accuracy to -1.0
    adam_stop = False
    stop_training = 20

    span_metric = NonBioSpanBasedF1Measure(
        idx_to_roles,
        tag_namespace="labels",
        ignore_classes=["O", "*"],
        ontology_path=params.ontology_path,
    )

    evaluate_frames(
        srl,
        1000,
        test,
        val_acc_best,
        word_vec,
        roles_to_idx,
        roles_to_idx,
        idx_to_roles,
        idx_to_roles,
        params,
        crf,
        adam_stop,
        stop_training,
        pred_constraints,
        span_metric,
        frame_to_idx,
        eval_type="test",
        final_eval=True,
        gold_data_file=test_data_file,
    )

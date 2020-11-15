import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np

from models.gcn import GCNLayer

from models.bilinear_scorer import BilinearScorer

from models.custom_allennlp.stacked_alternating_lstm import StackedAlternatingLstm
from models.custom_allennlp.elmo import Elmo


class SRL(nn.Module):
    def __init__(
        self,
        hidden_dim,
        tagset_size,
        num_layers,
        w_c_vocab_size,
        c_c_vocab_size,
        use_syntax,
        eln,
        num_layers_top,
        use_elmo,
        use_bert,
        params,
        gpu_id=-1,
    ):
        super(SRL, self).__init__()
        if gpu_id > -1:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_layers_top = num_layers_top
        self.eln = eln
        self.use_elmo = use_elmo
        self.use_bert = use_bert
        self.params = params
        self.dropout = nn.Dropout(p=params.gcn_dropout)
        self.embedding_dropout = nn.Dropout(p=params.emb_dropout)

        if self.use_elmo:
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            fixed_dim = 1024

            self.elmo = Elmo(
                options_file, weight_file, 1, dropout=0, do_layer_norm=False
            )

            fixed_dim += 100

        elif self.use_bert:
            fixed_dim = 768

        else:
            fixed_dim = 100

        embedding_dim = self.params.emb_dim
        self.indicator_embeddings = nn.Embedding(2, embedding_dim)

        self.tagset_size = tagset_size
        self.use_syntax = use_syntax
        self.num_layers_top = num_layers_top

        gcn_type = GCNLayer

        if self.params.non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif self.params.non_linearity == "tanh":
            self.non_linearity = nn.Tanh()
        elif self.params.non_linearity == "leakyrelu":
            self.non_linearity = nn.LeakyReLU()
        elif self.params.non_linearity == "celu":
            self.non_linearity = nn.CELU()
        elif self.params.non_linearity == "selu":
            self.non_linearity = nn.SELU()
        else:
            raise NotImplementedError

        self.lstm = StackedAlternatingLstm(
            fixed_dim + embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            recurrent_dropout_probability=0.1,
        )

        if self.use_syntax:
            if num_layers_top > 0:
                self.lstm_top = StackedAlternatingLstm(
                    hidden_dim,
                    hidden_dim,
                    num_layers=num_layers_top,
                    recurrent_dropout_probability=0.1,
                )
                self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
                self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
                self.bilinear_scorer = BilinearScorer(
                    hidden_dim, tagset_size, params.bilinear_dropout
                )

            else:
                self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
                self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
                self.bilinear_scorer = BilinearScorer(
                    hidden_dim, tagset_size, params.bilinear_dropout
                )

            self.gcn_w_c = gcn_type(
                hidden_dim,
                hidden_dim,
                w_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=True,
                dropout=self.params.gcn_dropout,
                non_linearity=self.non_linearity,
                edge_dropout=self.params.edge_dropout,
            )

            self.gcn_c_w = gcn_type(
                hidden_dim,
                hidden_dim,
                w_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=True,
                dropout=self.params.gcn_dropout,
                non_linearity=self.non_linearity,
                edge_dropout=self.params.edge_dropout,
            )

            self.gcn_c_c = gcn_type(
                hidden_dim,
                hidden_dim,
                c_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=False,
                dropout=self.params.gcn_dropout,
                non_linearity=self.non_linearity,
                edge_dropout=self.params.edge_dropout,
            )

        else:
            self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
            self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear_scorer = BilinearScorer(
                hidden_dim, tagset_size, params.bilinear_dropout
            )

        if self.eln:
            self.layernorm = nn.LayerNorm(fixed_dim)

    def forward(
        self,
        sentence,
        predicate_flags,
        sent_mask,
        lengths,
        fixed_embs,
        constituents,
        GCN_w_c,
        GCN_c_w,
        GCN_c_c,
        mask_const_batch,
        predicate_index,
        elmo_character_ids,
        bert_embs,
    ):

        if self.use_elmo:
            embeds = self.elmo(elmo_character_ids)["elmo_representations"][0]
            if not self.params.elmo_proj:
                embeds = torch.cat([embeds, fixed_embs], dim=2)

        elif self.use_bert:
            embeds = bert_embs
        else:
            embeds = fixed_embs

        if self.eln:
            embeds = self.layernorm(embeds * sent_mask.unsqueeze(2))

        embeds = self.embedding_dropout(embeds)

        embeds = torch.cat(
            (embeds, self.indicator_embeddings(predicate_flags.long())), 2
        )

        b, t, e = embeds.data.shape

        sent_len = torch.sort(lengths, descending=True)[0]
        idx_sort = torch.argsort(-lengths)

        if self.use_gpu:
            embeds = embeds.index_select(
                0, Variable(torch.cuda.LongTensor(idx_sort.cuda()))
            )
        else:
            embeds = embeds.index_select(0, Variable(torch.LongTensor(idx_sort)))

        packed = pack_padded_sequence(embeds, sent_len, batch_first=True)
        lstm_out, _ = self.lstm(packed)

        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [b, t, h]

        # Un-sort by length
        idx_unsort = torch.argsort(idx_sort)
        if self.use_gpu:
            lstm_out = lstm_out.index_select(
                0, Variable(torch.cuda.LongTensor(idx_unsort.cuda()))
            )
        else:
            lstm_out = lstm_out.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        if self.use_syntax:

            # Here I must concatenate the constituents with the lstm_out
            gcn_in = torch.cat([lstm_out, constituents], dim=1)
            mask_all = torch.cat([sent_mask, mask_const_batch], dim=1)
            # Apply graph conv
            adj_arc_in_w_c, adj_arc_out_w_c, adj_lab_in_w_c, adj_lab_out_w_c, mask_in_w_c, mask_out_w_c, mask_loop_w_c = (
                GCN_w_c
            )

            adj_arc_in_c_w, adj_arc_out_c_w, adj_lab_in_c_w, adj_lab_out_c_w, mask_in_c_w, mask_out_c_w, mask_loop_c_w = (
                GCN_c_w
            )

            adj_arc_in_c_c, adj_arc_out_c_c, adj_lab_in_c_c, adj_lab_out_c_c, mask_in_c_c, mask_out_c_c, mask_loop_c_c = (
                GCN_c_c
            )

            gcn_out = self.gcn_w_c(
                gcn_in,
                adj_arc_in_w_c,
                adj_arc_out_w_c,
                adj_lab_in_w_c,
                adj_lab_out_w_c,
                mask_in_w_c,
                mask_out_w_c,
                mask_loop_w_c,
                mask_all,
            )

            gcn_out = self.gcn_c_c(
                gcn_out,
                adj_arc_in_c_c,
                adj_arc_out_c_c,
                adj_lab_in_c_c,
                adj_lab_out_c_c,
                mask_in_c_c,
                mask_out_c_c,
                mask_loop_c_c,
                mask_all,
            )

            gcn_out = self.gcn_c_w(
                gcn_out,
                adj_arc_in_c_w,
                adj_arc_out_c_w,
                adj_lab_in_c_w,
                adj_lab_out_c_w,
                mask_in_c_w,
                mask_out_c_w,
                mask_loop_c_w,
                mask_all,
            )

            # Take back the lstm out only
            lstm_out = gcn_out.narrow(1, 0, t)
            if self.num_layers_top > 0:
                if self.use_gpu:
                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.cuda.LongTensor(idx_sort.cuda()))
                    )
                else:
                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.LongTensor(idx_sort))
                    )

                packed = pack_padded_sequence(lstm_out, sent_len, batch_first=True)
                lstm_out_, _ = self.lstm_top(packed)

                lstm_out_, _ = pad_packed_sequence(
                    lstm_out_, batch_first=True
                )  # [b, t, h]

                # Un-sort by length
                if self.use_gpu:
                    lstm_out_ = lstm_out_.index_select(
                        0, Variable(torch.cuda.LongTensor(idx_unsort.cuda()))
                    )
                else:
                    lstm_out_ = lstm_out_.index_select(
                        0, Variable(torch.LongTensor(idx_unsort))
                    )
                lstm_out = lstm_out_
        lstm_out_view = lstm_out.contiguous().view(b * t, -1)
        predicate_index = predicate_index.view(b * t)

        predicates_repr = lstm_out_view.index_select(0, predicate_index).view(b, t, -1)

        pred_repr = self.non_linearity(
            self.hidden2predicate(self.dropout(predicates_repr))
        )
        arg_repr = self.non_linearity(self.hidden2argument(self.dropout(lstm_out)))
        tag_scores = self.bilinear_scorer(pred_repr, arg_repr)  # [b*t, label_size]

        return tag_scores.view(b, t, self.tagset_size)


class SRL_Framenet(nn.Module):
    def __init__(
        self,
        hidden_dim,
        tagset_size,
        num_layers,
        w_c_vocab_size,
        c_c_vocab_size,
        use_syntax,
        eln,
        num_layers_top,
        params,
        gpu_id=-1,
    ):
        super(SRL_Framenet, self).__init__()

        if gpu_id > -1:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_layers_top = num_layers_top
        self.eln = eln
        self.params = params
        self.dropout = nn.Dropout(p=params.gcn_dropout)

        fixed_dim = 100

        embedding_dim = self.params.emb_dim
        self.indicator_embeddings = nn.Embedding(2, embedding_dim)

        self.tagset_size = tagset_size
        self.use_syntax = use_syntax
        self.num_layers_top = num_layers_top

        gcn_type = GCNLayer

        self.lstm = StackedAlternatingLstm(
            fixed_dim + embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            recurrent_dropout_probability=0.1,
        )

        if self.use_syntax:
            if num_layers_top > 0:
                if params.alter_top:
                    self.lstm_top = StackedAlternatingLstm(
                        hidden_dim,
                        hidden_dim,
                        num_layers=num_layers_top,
                        recurrent_dropout_probability=0.1,
                    )
                    self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
                    self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
                    self.bilinear_scorer = BilinearScorer(
                        hidden_dim, tagset_size, params.bilinear_dropout
                    )
                else:
                    self.lstm_top = nn.LSTM(
                        hidden_dim,
                        hidden_dim,
                        num_layers=num_layers_top,
                        batch_first=True,
                        bidirectional=True,
                        dropout=self.params.gcn_dropout,
                    )

                    self.hidden2predicate = nn.Linear(2 * hidden_dim, hidden_dim)
                    self.hidden2argument = nn.Linear(2 * hidden_dim, hidden_dim)
                    self.bilinear_scorer = BilinearScorer(
                        hidden_dim, tagset_size, params.bilinear_dropout
                    )
            else:
                self.hidden2predicate = nn.Linear(2 * hidden_dim, hidden_dim)
                self.hidden2argument = nn.Linear(2 * hidden_dim, hidden_dim)
                self.bilinear_scorer = BilinearScorer(
                    hidden_dim, tagset_size, params.bilinear_dropout
                )

            self.gcn_w_c = gcn_type(
                hidden_dim,
                hidden_dim,
                w_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=True,
                dropout=self.params.gcn_dropout,
            )

            self.gcn_c_w = gcn_type(
                hidden_dim,
                hidden_dim,
                w_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=True,
                dropout=self.params.gcn_dropout,
            )

            self.gcn_c_c = gcn_type(
                hidden_dim,
                hidden_dim,
                c_c_vocab_size,
                in_arcs=True,
                out_arcs=True,
                use_gates=True,
                batch_first=True,
                residual=True,
                no_loop=False,
                dropout=self.params.gcn_dropout,
            )

        else:
            self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
            self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear_scorer = BilinearScorer(
                hidden_dim, tagset_size, params.bilinear_dropout
            )

        if self.eln:
            self.layernorm = nn.LayerNorm(fixed_dim)

    def forward(
        self,
        sentence,
        predicate_flags,
        sent_mask,
        lengths,
        fixed_embs,
        constituents,
        GCN_w_c,
        GCN_c_w,
        GCN_c_c,
        mask_const_batch,
        predicate_index,
        softmax_constraints,
        frame_emb_batch,
    ):

        embeds = fixed_embs

        if self.eln:
            embeds = self.layernorm(embeds * sent_mask.unsqueeze(2))

        if self.params.emb_dropout:
            embeds = self.dropout(embeds)

        embeds = torch.cat(
            (embeds, self.indicator_embeddings(predicate_flags.long())), 2
        )

        b, t, e = embeds.data.shape

        # Sort by length (keep idx)
        sent_len = torch.sort(lengths, descending=True)[0]
        idx_sort = torch.argsort(-lengths)

        if self.use_gpu:
            embeds = embeds.index_select(
                0, Variable(torch.cuda.LongTensor(idx_sort.cuda()))
            )
        else:
            embeds = embeds.index_select(0, Variable(torch.LongTensor(idx_sort)))

        packed = pack_padded_sequence(embeds, sent_len, batch_first=True)

        lstm_out, _ = self.lstm(packed)

        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [b, t, h]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        if self.use_gpu:
            lstm_out = lstm_out.index_select(
                0, Variable(torch.cuda.LongTensor(idx_unsort.cuda()))
            )
        else:
            lstm_out = lstm_out.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        if self.use_syntax:
            # Here I must concatenate the constituents with the lstm_out
            gcn_in = torch.cat([lstm_out, constituents], dim=1)
            mask_all = torch.cat([sent_mask, mask_const_batch], dim=1)
            # Apply graph conv
            adj_arc_in_w_c, adj_arc_out_w_c, adj_lab_in_w_c, adj_lab_out_w_c, mask_in_w_c, mask_out_w_c, mask_loop_w_c = (
                GCN_w_c
            )

            adj_arc_in_c_w, adj_arc_out_c_w, adj_lab_in_c_w, adj_lab_out_c_w, mask_in_c_w, mask_out_c_w, mask_loop_c_w = (
                GCN_c_w
            )

            adj_arc_in_c_c, adj_arc_out_c_c, adj_lab_in_c_c, adj_lab_out_c_c, mask_in_c_c, mask_out_c_c, mask_loop_c_c = (
                GCN_c_c
            )

            gcn_out = self.gcn_w_c(
                gcn_in,
                adj_arc_in_w_c,
                adj_arc_out_w_c,
                adj_lab_in_w_c,
                adj_lab_out_w_c,
                mask_in_w_c,
                mask_out_w_c,
                mask_loop_w_c,
                mask_all,
            )

            gcn_out = self.gcn_c_c(
                gcn_out,
                adj_arc_in_c_c,
                adj_arc_out_c_c,
                adj_lab_in_c_c,
                adj_lab_out_c_c,
                mask_in_c_c,
                mask_out_c_c,
                mask_loop_c_c,
                mask_all,
            )

            gcn_out = self.gcn_c_w(
                gcn_out,
                adj_arc_in_c_w,
                adj_arc_out_c_w,
                adj_lab_in_c_w,
                adj_lab_out_c_w,
                mask_in_c_w,
                mask_out_c_w,
                mask_loop_c_w,
                mask_all,
            )

            # Take back the lstm out only
            lstm_out = gcn_out.narrow(1, 0, t)
            if self.num_layers_top > 0:
                if self.use_gpu:
                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.cuda.LongTensor(idx_sort.cuda()))
                    )
                else:

                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.LongTensor(idx_sort))
                    )

                packed = pack_padded_sequence(lstm_out, sent_len, batch_first=True)

                lstm_out, _ = self.lstm_top(packed)

                lstm_out, _ = pad_packed_sequence(
                    lstm_out, batch_first=True
                )  # [b, t, h]

                # Un-sort by length
                if self.use_gpu:
                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.cuda.LongTensor(idx_unsort.cuda()))
                    )
                else:
                    lstm_out = lstm_out.index_select(
                        0, Variable(torch.LongTensor(idx_unsort))
                    )

        lstm_out_view = lstm_out.contiguous().view(b * t, -1)
        predicate_index = predicate_index.view(b * t)

        predicates_repr = lstm_out_view.index_select(0, predicate_index).view(b, t, -1)

        pred_repr = F.relu(self.hidden2predicate(self.dropout(predicates_repr)))
        arg_repr = F.relu(self.hidden2argument(self.dropout(lstm_out)))
        tag_scores = self.bilinear_scorer(pred_repr, arg_repr)  # [b*t, label_size]

        tag_scores = tag_scores.view(b, t, -1)
        tag_scores = tag_scores.masked_fill(
            (1 - softmax_constraints.view(b, 1, -1)).byte(), float("-1e13")
        )  # 1e-13)

        tag_scores = tag_scores.view(b * t, -1)

        return tag_scores.view(b, t, self.tagset_size)

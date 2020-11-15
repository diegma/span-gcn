import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class GCNLayer(nn.Module):
    """ Convolutional Neural Network Layer
    """

    def __init__(
        self,
        num_inputs,
        num_units,
        num_labels,
        dropout=0.0,
        in_arcs=True,
        out_arcs=True,
        batch_first=False,
        use_gates=True,
        residual=False,
        no_loop=False,
        non_linearity="relu",
        edge_dropout=0.0,
    ):
        super(GCNLayer, self).__init__()

        self.init_gcn(
            batch_first,
            dropout,
            edge_dropout,
            in_arcs,
            no_loop,
            non_linearity,
            num_inputs,
            num_labels,
            num_units,
            out_arcs,
            residual,
            use_gates,
        )

        if in_arcs:
            self.V_in = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_in)

            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_in, 0)

            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_in_gate, 1)

        if out_arcs:
            # self.V_out = autograd.Variable(torch.FloatTensor(self.num_inputs, self.num_units))
            self.V_out = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_out)

            # self.b_out = autograd.Variable(torch.FloatTensor(num_labels, self.num_units))
            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_out, 0)

            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_out_gate, 1)
        if not self.no_loop:
            self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.W_self_loop)

            if self.use_gates:
                self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.W_self_loop_gate)

    def init_gcn(
        self,
        batch_first,
        dropout,
        edge_dropout,
        in_arcs,
        no_loop,
        non_linearity,
        num_inputs,
        num_labels,
        num_units,
        out_arcs,
        residual,
        use_gates,
    ):
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.no_loop = no_loop
        self.retain = 1.0 - edge_dropout
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.non_linearity = non_linearity
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(num_units)

    # @profile
    def forward(
        self,
        src,
        arc_tensor_in=None,
        arc_tensor_out=None,
        label_tensor_in=None,
        label_tensor_out=None,
        mask_in=None,
        mask_out=None,  # batch* t, degree
        mask_loop=None,
        sent_mask=None,
    ):

        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()

        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view(
            (batch_size * seq_len, self.num_inputs)
        )  # [b* t, h]
        input_ = self.dropout(input_)
        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in.index_select(
                0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
            )  # [b* t* degr, h]
            second_in = self.b_in.index_select(0, label_tensor_in[0])  # [b* t* degr, h]
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size // seq_len)
            in_ = in_.view((batch_size, seq_len, degr, self.num_units))
            if self.use_gates:
                # compute gate weights
                input_in_gate = torch.mm(
                    input_, self.V_in_gate
                )  # [b* t, h] * [h,h] = [b*t, h]
                first_in_gate = input_in_gate.index_select(
                    0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
                )  # [b* t* mxdeg, h]
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view(
                    (batch_size, seq_len, degr)
                )

            max_degree += degr

        if self.out_arcs:
            input_out = torch.mm(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            first_out = input_out.index_select(
                0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
            )  # [b* t* mxdeg, h]
            second_out = self.b_out.index_select(0, label_tensor_out[0])

            degr = int(first_out.size()[0] / batch_size // seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view(
                (batch_size, seq_len, degr, self.num_units)
            )

            if self.use_gates:
                # compute gate weights
                input_out_gate = torch.mm(
                    input_, self.V_out_gate
                )  # [b* t, h] * [h,h] = [b* t, h]
                first_out_gate = input_out_gate.index_select(
                    0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
                )  # [b* t* mxdeg, h]
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view(
                    (batch_size, seq_len, degr)
                )
        if self.no_loop:
            if self.in_arcs and self.out_arcs:
                potentials = torch.cat((in_, out_), dim=2)  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_out), dim=1)  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = out_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = out_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_out  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = in_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = in_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_in  # [b* t, mxdeg]
            max_degree -= 1
        else:
            same_input = torch.mm(input_, self.W_self_loop).view(
                encoder_outputs.size(0), encoder_outputs.size(1), -1
            )
            same_input = same_input.view(
                encoder_outputs.size(0),
                encoder_outputs.size(1),
                1,
                self.W_self_loop.size(1),
            )
            if self.use_gates:
                same_input_gate = torch.mm(input_, self.W_self_loop_gate).view(
                    encoder_outputs.size(0), encoder_outputs.size(1), -1
                )

            if self.in_arcs and self.out_arcs:
                potentials = torch.cat(
                    (in_, out_, same_input), dim=2
                )  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat(
                    (mask_in, mask_out, mask_loop), dim=1
                )  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = torch.cat(
                    (out_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = torch.cat(
                    (in_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]
            else:
                potentials = same_input  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = same_input_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_loop  # [b* t, mxdeg]

        potentials_resh = potentials.view(
            (batch_size * seq_len, max_degree, self.num_units)
        )  # [h, b * t, mxdeg]

        if self.use_gates:
            potentials_r = potentials_gate.view(
                (batch_size * seq_len, max_degree)
            )  # [b * t, mxdeg]
            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(
                2
            )  # [b * t, mxdeg]

            potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        else:
            # NO Gates
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)

        if self.retain == 1 or not self.training:
            pass
        else:
            mat_1 = torch.Tensor(mask_soft.data.size()).uniform_(0, 1)
            ret = torch.Tensor([self.retain])
            mat_2 = (mat_1 < ret).float()
            drop_mask = Variable(mat_2, requires_grad=False)
            if potentials_resh.is_cuda:
                drop_mask = drop_mask.cuda()

            potentials_masked *= drop_mask.unsqueeze(2)

        potentials_masked_ = potentials_masked.sum(dim=1)  # [b * t, h]

        potentials_masked_ = self.layernorm(potentials_masked_) * sent_mask.view(
            batch_size * seq_len
        ).unsqueeze(1)

        potentials_masked_ = self.non_linearity(potentials_masked_)  # [b * t, h]

        result_ = potentials_masked_.view(
            (batch_size, seq_len, self.num_units)
        )  # [ b, t, h]

        result_ = result_ * sent_mask.unsqueeze(2)  # [b, t, h]
        memory_bank = result_  # [t, b, h]

        if self.residual:
            memory_bank += src

        return memory_bank


def get_adj_BE(batch, max_batch_len, gpu_id, max_degr_in, max_degr_out, forward):
    node1_index = [[word[1] for word in sent] for sent in batch]
    node2_index = [[word[2] for word in sent] for sent in batch]
    label_index = [[word[0] for word in sent] for sent in batch]
    begin_index = [[word[3] for word in sent] for sent in batch]

    batch_size = len(batch)

    _MAX_BATCH_LEN = max_batch_len
    _MAX_DEGREE_IN = max_degr_in
    _MAX_DEGREE_OUT = max_degr_out

    adj_arc_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 2), dtype="int32"
    )
    adj_lab_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 1), dtype="int32"
    )
    adj_arc_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 2), dtype="int32"
    )
    adj_lab_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 1), dtype="int32"
    )

    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN), dtype="float32")
    mask_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT), dtype="float32"
    )
    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype="float32")

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(node1_index):  # iterates over the batch
        for a, arc in enumerate(de):
            if not forward:
                arc_1 = arc
                arc_2 = node2_index[d][a]
            else:
                arc_2 = arc
                arc_1 = node2_index[d][a]

            if begin_index[d][a] == 0:  # BEGIN
                if arc_1 in tmp_in:
                    tmp_in[arc_1] += 1
                else:
                    tmp_in[arc_1] = 0

                idx_in = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_IN)
                    + arc_1 * _MAX_DEGREE_IN
                    + tmp_in[arc_1]
                )

                if tmp_in[arc_1] < _MAX_DEGREE_IN:
                    adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                    adj_lab_in[idx_in] = np.array([label_index[d][a]])  # incoming arcs
                    mask_in[idx_in] = 1.0

            else:  # END
                if arc_1 in tmp_out:
                    tmp_out[arc_1] += 1
                else:
                    tmp_out[arc_1] = 0

                idx_out = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_OUT)
                    + arc_1 * _MAX_DEGREE_OUT
                    + tmp_out[arc_1]
                )

                if tmp_out[arc_1] < _MAX_DEGREE_OUT:
                    adj_arc_out[idx_out] = np.array([d, arc_2])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array(
                        [label_index[d][a]]
                    )  # outgoing arcs
                    mask_out[idx_out] = 1.0

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = torch.LongTensor(np.transpose(adj_arc_in).tolist())
    adj_arc_out = torch.LongTensor(np.transpose(adj_arc_out).tolist())

    adj_lab_in = torch.LongTensor(np.transpose(adj_lab_in).tolist())
    adj_lab_out = torch.LongTensor(np.transpose(adj_lab_out).tolist())

    mask_in = autograd.Variable(
        torch.FloatTensor(
            mask_in.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_IN)).tolist()
        ),
        requires_grad=False,
    )
    mask_out = autograd.Variable(
        torch.FloatTensor(
            mask_out.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_OUT)).tolist()
        ),
        requires_grad=False,
    )
    mask_loop = autograd.Variable(
        torch.FloatTensor(mask_loop.tolist()), requires_grad=False
    )

    if gpu_id > -1:
        adj_arc_in = adj_arc_in.cuda()
        adj_arc_out = adj_arc_out.cuda()
        adj_lab_in = adj_lab_in.cuda()
        adj_lab_out = adj_lab_out.cuda()
        mask_in = mask_in.cuda()
        mask_out = mask_out.cuda()
        mask_loop = mask_loop.cuda()
    return [
        adj_arc_in,
        adj_arc_out,
        adj_lab_in,
        adj_lab_out,
        mask_in,
        mask_out,
        mask_loop,
    ]

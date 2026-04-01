"""
SCPN — Syntactically Controlled Paraphrase Network

Iyyer et al., 2018 (NAACL, arXiv:1804.06059)

Exact reimplementation of the original SCPN encoder-decoder model
from github.com/miyyer/scpn, ported to modern Python 3 / PyTorch.

Architecture (per Section 3 of the paper):
  - Bidirectional LSTM encoder for the input sentence
  - Unidirectional LSTM encoder for the target parse tree
  - 2-layer LSTM decoder with:
    - Bilinear attention over encoder hidden states
    - Bilinear attention over parse hidden states (Section 3.1, Eq. 1)
    - Copy mechanism (See et al., 2017)
  - BPE tokenization (subword-nmt)
  - A separate ParseNet that generates full parse trees from
    top-2-level templates (Section 3.2)

Pipeline (exact match to generate_paraphrases.py):
  1. Constituency-parse the input sentence
  2. Linearize the parse tree (remove leaf tokens)
  3. For each target template, use ParseNet to generate a full parse
  4. BPE-segment the input sentence
  5. Feed sentence + full parse to SCPN encoder-decoder
  6. Reverse BPE to get final paraphrase text
  7. Filter by n-gram overlap (≥0.5) and semantic similarity (≥0.7)
  8. Query victim model and select adversarial paraphrase

References:
  Paper — https://aclanthology.org/N18-1170/
  Code  — https://github.com/miyyer/scpn
"""

import logging
import os
import pickle
import tarfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

logger = logging.getLogger("textattack.attacks.scpn")


# ── Constants ──────────────────────────────────────────────────────────

_CACHE_DIR = Path.home() / ".cache" / "scpn"
_GITHUB_RAW = "https://raw.githubusercontent.com/miyyer/scpn/master"
_GDRIVE_MODELS_ID = "1AuH1aHrE9maYttuSJz_9eltYOAad8Mfj"

# 10 most frequent templates from the original generate_paraphrases.py.
# These are the top-2-level parse templates used in the paper (Section 3.3):
# "selecting the twenty most frequent templates in PARANMT-50M"
SCPN_TEMPLATES = [
    "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP",
    "( ROOT ( S ( VP ) ( . ) ) ) EOP",
    "( ROOT ( NP ( NP ) ( . ) ) ) EOP",
    "( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP",
    "( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP",
    "( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP",
    "( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP",
    "( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP",
    "( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP",
    "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP",
]

# PTB tagset — embedded from miyyer/scpn/data/ptb_tagset.txt.
# Order matters: index positions must match the pretrained model weights.
PTB_TAGSET = [
    "PAD", "ROOT", "ADJP", "-ADV", "ADVP", "-BNF", "CC", "CD", "-CLF",
    "-CLR", "CONJP", "-DIR", "DT", "-DTV", "EX", "-EXT", "FRAG", "FW",
    "-HLN", "IN", "INTJ", "JJ", "JJR", "JJS", "-LGS", "-LOC", "LS",
    "LST", "MD", "-MNR", "NAC", "NN", "NNS", "NNP", "NNPS", "-NOM",
    "NP", "NX", "PDT", "POS", "PP", "-PRD", "PRN", "PRP", "-PRP",
    "PRP$", "PRP-S", "PRT", "-PUT", "QP", "RB", "RBR", "RBS", "RP",
    "RRC", "S", "SBAR", "SBARQ", "-SBJ", "SINV", "SQ", "SYM", "-TMP",
    "TO", "-TPC", "-TTL", "UCP", "UH", "VB", "VBD", "VBG", "VBN",
    "VBP", "VBZ", "-VOC", "VP", "WDT", "WHADJP", "WHADVP", "WHNP",
    "WHPP", "WP", "WP$", "WP-S", "WRB", "X", "#", "``", "''", ".",
    ",", ":", "$", "-LRB-", "-RRB-", "-NONE-", "*", "0", "T", "NUL",
    ">", "(", ")", "EOP",
]

PARSE_GEN_VOC = {tag: idx for idx, tag in enumerate(PTB_TAGSET)}
REV_PARSE_GEN_VOC = {idx: tag for tag, idx in PARSE_GEN_VOC.items()}


# ── Utility Functions (exact ports from scpn_utils.py / generate_paraphrases.py) ──

def _is_paren(tok):
    return tok == ")" or tok == "("


def _deleaf(tree):
    """Remove leaf nodes from a parse tree, returning the linearized
    nonterminal skeleton plus an EOP sentinel.

    Exact port of ``deleaf()`` from ``scpn_utils.py``."""
    tree_str = str(tree).replace("\n", "")
    nonleaves = ""
    for w in tree_str.split():
        w = w.replace("(", "( ").replace(")", " )")
        nonleaves += w + " "

    arr = nonleaves.split()
    for n in range(len(arr)):
        if n + 1 < len(arr):
            if not _is_paren(arr[n]) and not _is_paren(arr[n + 1]):
                arr[n + 1] = ""

    return " ".join(arr).split() + ["EOP"]


def _reverse_bpe(tokens):
    """Reverse BPE segmentation.

    Exact port from ``generate_paraphrases.py``."""
    x = []
    cache = ""
    for w in tokens:
        if w.endswith("@@"):
            cache += w.replace("@@", "")
        elif cache != "":
            x.append(cache + w)
            cache = ""
        else:
            x.append(w)
    return " ".join(x)


def _ngram_overlap(text_a, text_b, n=4):
    """N-gram overlap ratio between two texts.

    Paper Section 3.3: "we set minimum n-gram overlap to 0.5"."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()

    if len(words_a) < n or len(words_b) < n:
        set_a, set_b = set(words_a), set(words_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))

    ngrams_a = {tuple(words_a[i : i + n]) for i in range(len(words_a) - n + 1)}
    ngrams_b = {tuple(words_b[i : i + n]) for i in range(len(words_b) - n + 1)}
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / max(len(ngrams_a), len(ngrams_b))


# ── SCPN Model Architecture ───────────────────────────────────────────
# Exact port of the ``SCPN`` class from miyyer/scpn/train_scpn.py.
# "seq2seq w/ decoder attention; transformation embeddings concatenated
#  with decoder word inputs; attention conditioned on transformation
#  via bilinear product" — original source header.

class SCPNModel(nn.Module):

    def __init__(self, d_word, d_hid, d_nt, d_trans,
                 len_voc, len_trans_voc, use_input_parse):
        super().__init__()
        self.d_word = d_word
        self.d_hid = d_hid
        self.d_trans = d_trans
        self.d_nt = d_nt + 1
        self.len_voc = len_voc
        self.len_trans_voc = len_trans_voc
        self.use_input_parse = use_input_parse

        self.word_embs = nn.Embedding(len_voc, d_word)
        self.trans_embs = nn.Embedding(len_trans_voc, self.d_nt)

        enc_input = d_word + d_trans if use_input_parse else d_word
        self.encoder = nn.LSTM(
            enc_input, d_hid, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        self.encoder_proj = nn.Linear(d_hid * 2, d_hid)
        self.decoder = nn.LSTM(
            d_word + d_hid, d_hid, num_layers=2, batch_first=True,
        )
        self.trans_encoder = nn.LSTM(
            self.d_nt, d_trans, num_layers=1, batch_first=True,
        )

        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)

        self.att_parse_proj = nn.Linear(d_trans, d_hid)
        self.att_W = nn.Parameter(torch.empty(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.empty(d_hid, d_hid))
        nn.init.xavier_uniform_(self.att_W)
        nn.init.xavier_uniform_(self.att_parse_W)

        self.copy_hid_v = nn.Parameter(torch.empty(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.empty(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.empty(d_word + d_hid, 1))
        nn.init.xavier_uniform_(self.copy_hid_v)
        nn.init.xavier_uniform_(self.copy_att_v)
        nn.init.xavier_uniform_(self.copy_inp_v)

        self.register_buffer(
            "_e_hid_init", torch.zeros(2, 1, d_hid), persistent=False)
        self.register_buffer(
            "_e_cell_init", torch.zeros(2, 1, d_hid), persistent=False)
        self.register_buffer(
            "_d_cell_init", torch.zeros(2, 1, d_hid), persistent=False)
        self.register_buffer(
            "_t_hid_init", torch.zeros(1, 1, d_trans), persistent=False)
        self.register_buffer(
            "_t_cell_init", torch.zeros(1, 1, d_trans), persistent=False)

    def _mask(self, lengths):
        mx = torch.max(lengths)
        rng = torch.arange(mx, device=lengths.device).unsqueeze(0)
        return (rng < lengths.unsqueeze(1)).float()

    def _masked_softmax(self, scores, mask):
        s = torch.nn.functional.softmax(scores, dim=-1) * mask
        return s / (s.sum(dim=1, keepdim=True) + 1e-13)

    def _attn_encoder(self, h_prev, enc_hids, lens):
        mask = self._mask(lens)
        proj = h_prev.mm(self.att_W)
        scores = (proj.unsqueeze(1) * enc_hids).sum(2)
        return self._masked_softmax(scores, mask)

    def _attn_parse(self, h_prev, parse_hids, lens):
        mask = self._mask(lens)
        proj = h_prev.mm(self.att_parse_W)
        scores = (proj.unsqueeze(1) * parse_hids).sum(2)
        return self._masked_softmax(scores, mask)

    def encode_batch(self, inputs, trans, lengths):
        bsz, max_len = inputs.size()
        embs = self.word_embs(inputs)
        lens, idx = torch.sort(lengths, 0, True)
        if self.use_input_parse and trans is not None:
            embs = torch.cat(
                [embs, trans.unsqueeze(1).expand(bsz, max_len, self.d_trans)],
                2,
            )
        h0 = self._e_hid_init.expand(2, bsz, self.d_hid).contiguous()
        c0 = self._e_cell_init.expand(2, bsz, self.d_hid).contiguous()
        packed, (last_h, _) = self.encoder(
            pack(embs[idx], lens.tolist(), batch_first=True), (h0, c0),
        )
        _, rev = torch.sort(idx, 0)
        all_h = unpack(packed, batch_first=True)[0][rev]
        all_h = self.encoder_proj(
            all_h.reshape(-1, self.d_hid * 2),
        ).view(bsz, max_len, self.d_hid)
        last_h = torch.cat([last_h[0], last_h[1]], 1)
        last_h = self.encoder_proj(last_h)[rev]
        return all_h, last_h

    def encode_transformations(self, trans, lengths, return_last=True):
        bsz, _ = trans.size()
        lens, idx = torch.sort(lengths, 0, True)
        embs = self.trans_embs(trans)
        h0 = self._t_hid_init.expand(1, bsz, self.d_trans).contiguous()
        c0 = self._t_cell_init.expand(1, bsz, self.d_trans).contiguous()
        packed, (last_h, _) = self.trans_encoder(
            pack(embs[idx], lens.tolist(), batch_first=True), (h0, c0),
        )
        _, rev = torch.sort(idx, 0)
        if return_last:
            return last_h.squeeze(0)[rev]
        return unpack(packed, batch_first=True)[0][rev]

    def _decode_step(self, step, prev_words, prev_h, prev_c,
                     enc_hids, parse_hids, enc_lens, parse_lens, bsz):
        if step == 0:
            w_inp = torch.zeros(bsz, 1, self.d_word, device=enc_hids.device)
        else:
            w_inp = self.word_embs(prev_words).view(bsz, 1, self.d_word)

        tw = self._attn_parse(prev_h[1], parse_hids, parse_lens)
        t_ctx = (tw.unsqueeze(2) * parse_hids).sum(1)
        dec_inp = torch.cat([w_inp, t_ctx.unsqueeze(1)], 2)

        _, (hn, cn) = self.decoder(dec_inp, (prev_h, prev_c))

        aw = self._attn_encoder(hn[1], enc_hids, enc_lens)
        a_ctx = (aw.unsqueeze(2) * enc_hids).sum(1)

        p_copy = (dec_inp.squeeze(1).mm(self.copy_inp_v)
                  + a_ctx.mm(self.copy_att_v)
                  + hn[1].mm(self.copy_hid_v))
        p_copy = torch.sigmoid(p_copy).squeeze(1)
        return hn, cn, aw, a_ctx, p_copy

    @torch.no_grad()
    def batch_beam_search(self, inputs, out_trans, in_sent_lens,
                          out_trans_lens, eos_idx, beam_size=5, max_steps=70):
        bsz, max_len = inputs.size()
        inputs = inputs[:, :in_sent_lens[0]]

        ot_hids = self.encode_transformations(
            out_trans, out_trans_lens, return_last=False,
        )
        ot_hids = self.att_parse_proj(ot_hids)
        enc_hids, enc_last = self.encode_batch(inputs, None, in_sent_lens)

        hn = enc_last.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self._d_cell_init

        beam_dict: OrderedDict[int, list] = OrderedDict()
        for b_idx in range(out_trans.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        while True:
            prev_words: list | None = []
            prev_hs: list = []
            prev_cs: list = []

            for b_idx in beam_dict:
                for b in beam_dict[b_idx]:
                    _, ph, pc, seq = b
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None
                    prev_hs.append(ph)
                    prev_cs.append(pc)

            hs = torch.cat(prev_hs, 1)
            cs = torch.cat(prev_cs, 1)
            ne = hs.size(1)
            if prev_words is not None:
                prev_words = torch.tensor(
                    prev_words, dtype=torch.long, device=inputs.device,
                )

            if ne != ot_hids.size(0):
                d1, d2, d3 = ot_hids.size()
                rf = ne // d1
                cur_ot = ot_hids.unsqueeze(1).expand(
                    d1, rf, d2, d3).contiguous().view(-1, d2, d3)
                cur_otl = out_trans_lens.unsqueeze(1).expand(
                    d1, rf).contiguous().view(-1)
            else:
                cur_ot, cur_otl = ot_hids, out_trans_lens

            _, il, hd = enc_hids.size()
            cur_eh = enc_hids.expand(ne, il, hd)
            cur_el = in_sent_lens.expand(ne)
            cur_inp = inputs.expand(ne, in_sent_lens[0])

            hn, cn, aw, a_ctx, p_copy = self._decode_step(
                nsteps, prev_words, hs, cs,
                cur_eh, cur_ot, cur_el, cur_otl, ne,
            )

            v_scores = torch.zeros(ne, self.len_voc, device=inputs.device)
            v_scores.scatter_add_(1, cur_inp, aw)
            v_scores = torch.log(v_scores + 1e-20)

            pred = torch.log_softmax(
                self.out_dense_2(self.out_dense_1(
                    torch.cat([hn[1], a_ctx], 1))),
                dim=-1,
            )
            fp = p_copy.unsqueeze(1) * v_scores + (1 - p_copy.unsqueeze(1)) * pred

            for b_idx in beam_dict:
                cands = []
                if ne == len(beam_dict):
                    ex_hn = hn[:, b_idx, :].unsqueeze(1)
                    ex_cn = cn[:, b_idx, :].unsqueeze(1)
                    ep = fp[b_idx]
                    _, ti = torch.sort(-ep)
                    for z in range(beam_size):
                        wi = ti[z].item()
                        cands.append((ep[wi].item(), ex_hn, ex_cn, [wi]))
                    beam_dict[b_idx] = cands
                else:
                    obs = beam_dict[b_idx]
                    s = b_idx * beam_size
                    e = s + beam_size
                    ex_hn = hn[:, s:e, :]
                    ex_cn = cn[:, s:e, :]
                    ex_fp = fp[s:e]
                    for oi, ob in enumerate(obs):
                        cp, _, _, seq = ob
                        if seq[-1] == eos_idx:
                            cands.append(ob)
                        op = ex_fp[oi]
                        oh = ex_hn[:, oi, :].unsqueeze(1)
                        oc = ex_cn[:, oi, :].unsqueeze(1)
                        _, ti = torch.sort(-op)
                        for z in range(beam_size):
                            wi = ti[z].item()
                            cands.append((
                                cp + float(op[wi].cpu().item()),
                                oh, oc, seq + [wi],
                            ))
                    si = np.argsort([x[0] for x in cands])[::-1]
                    beam_dict[b_idx] = [cands[x] for x in si][:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict


# ── ParseNet Model Architecture ───────────────────────────────────────
# Exact port of ``ParseNet`` from miyyer/scpn/train_parse_generator.py.
# Generates full constituency parses from top-2-level templates
# (Section 3.2 of the paper).

class ParseNetModel(nn.Module):

    def __init__(self, d_nt, d_hid, len_voc):
        super().__init__()
        self.d_nt = d_nt
        self.d_hid = d_hid
        self.len_voc = len_voc

        self.trans_embs = nn.Embedding(len_voc, d_nt)
        self.encoder = nn.LSTM(d_nt, d_hid, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(
            d_nt + d_hid, d_hid, num_layers=1, batch_first=True,
        )

        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)

        self.att_W = nn.Parameter(torch.empty(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.empty(d_hid, d_hid))
        nn.init.xavier_uniform_(self.att_W)
        nn.init.xavier_uniform_(self.att_parse_W)

        self.copy_hid_v = nn.Parameter(torch.empty(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.empty(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.empty(d_nt + d_hid, 1))
        nn.init.xavier_uniform_(self.copy_hid_v)
        nn.init.xavier_uniform_(self.copy_att_v)
        nn.init.xavier_uniform_(self.copy_inp_v)

        self.register_buffer(
            "_e_hid_init", torch.zeros(1, 1, d_hid), persistent=False)
        self.register_buffer(
            "_e_cell_init", torch.zeros(1, 1, d_hid), persistent=False)
        self.register_buffer(
            "_d_cell_init", torch.zeros(1, 1, d_hid), persistent=False)

    def _mask(self, lengths):
        mx = torch.max(lengths)
        rng = torch.arange(mx, device=lengths.device).unsqueeze(0)
        return (rng < lengths.unsqueeze(1)).float()

    def _masked_softmax(self, scores, mask):
        s = torch.nn.functional.softmax(scores, dim=-1) * mask
        return s / (s.sum(dim=1, keepdim=True) + 1e-13)

    def _attn_enc(self, h_prev, enc_hids, lens):
        mask = self._mask(lens)
        proj = h_prev[0].mm(self.att_W)
        scores = (proj.unsqueeze(1) * enc_hids).sum(2)
        return self._masked_softmax(scores, mask)

    def _attn_tmpl(self, h_prev, tmpl_hids, lens):
        mask = self._mask(lens)
        proj = h_prev[0].mm(self.att_parse_W)
        scores = (proj.unsqueeze(1) * tmpl_hids).sum(2)
        return self._masked_softmax(scores, mask)

    def encode_batch(self, inputs, lengths):
        bsz, _ = inputs.size()
        embs = self.trans_embs(inputs)
        lens, idx = torch.sort(lengths, 0, True)
        h0 = self._e_hid_init.expand(1, bsz, self.d_hid).contiguous()
        c0 = self._e_cell_init.expand(1, bsz, self.d_hid).contiguous()
        packed, (last_h, _) = self.encoder(
            pack(embs[idx], lens.tolist(), batch_first=True), (h0, c0),
        )
        _, rev = torch.sort(idx, 0)
        all_h = unpack(packed, batch_first=True)[0]
        return all_h[rev], last_h.squeeze(0)[rev]

    def _decode_step(self, step, prev_words, prev_h, prev_c,
                     enc_hids, tmpl_hids, enc_lens, tmpl_lens, bsz):
        if step == 0:
            w_inp = torch.zeros(bsz, 1, self.d_nt, device=enc_hids.device)
        else:
            w_inp = self.trans_embs(prev_words).view(bsz, 1, self.d_nt)

        tw = self._attn_tmpl(prev_h, tmpl_hids, tmpl_lens)
        t_ctx = (tw.unsqueeze(2) * tmpl_hids).sum(1)
        dec_inp = torch.cat([w_inp, t_ctx.unsqueeze(1)], 2)

        _, (hn, cn) = self.decoder(dec_inp, (prev_h, prev_c))

        aw = self._attn_enc(hn, enc_hids, enc_lens)
        a_ctx = (aw.unsqueeze(2) * enc_hids).sum(1)

        p_copy = (dec_inp.squeeze(1).mm(self.copy_inp_v)
                  + a_ctx.mm(self.copy_att_v)
                  + hn.squeeze(0).mm(self.copy_hid_v))
        p_copy = torch.sigmoid(p_copy).squeeze(1)
        return hn, cn, aw, a_ctx, p_copy

    @torch.no_grad()
    def batch_beam_search(self, inputs, out_trimmed, in_trans_lens,
                          out_trimmed_lens, eos_idx,
                          beam_size=5, max_steps=250):
        bsz, max_len = inputs.size()
        inputs = inputs[:, :in_trans_lens[0]]

        enc_hids, enc_last = self.encode_batch(inputs, in_trans_lens)
        trim_hids, _ = self.encode_batch(out_trimmed, out_trimmed_lens)

        hn = enc_last.unsqueeze(0)
        cn = self._d_cell_init

        beam_dict: OrderedDict[int, list] = OrderedDict()
        for b_idx in range(trim_hids.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        while True:
            prev_words: list | None = []
            prev_hs: list = []
            prev_cs: list = []

            for b_idx in beam_dict:
                for b in beam_dict[b_idx]:
                    _, ph, pc, seq = b
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None
                    prev_hs.append(ph)
                    prev_cs.append(pc)

            hs = torch.cat(prev_hs, 1)
            cs = torch.cat(prev_cs, 1)
            ne = hs.size(1)
            if prev_words is not None:
                prev_words = torch.tensor(
                    prev_words, dtype=torch.long, device=inputs.device,
                )

            if ne != trim_hids.size(0):
                d1, d2, d3 = trim_hids.size()
                rf = ne // d1
                cur_th = trim_hids.unsqueeze(1).expand(
                    d1, rf, d2, d3).contiguous().view(-1, d2, d3)
                cur_tl = out_trimmed_lens.unsqueeze(1).expand(
                    d1, rf).contiguous().view(-1)
            else:
                cur_th, cur_tl = trim_hids, out_trimmed_lens

            _, il, hd = enc_hids.size()
            cur_eh = enc_hids.expand(ne, il, hd)
            cur_el = in_trans_lens.expand(ne)
            cur_inp = inputs.expand(ne, in_trans_lens[0])

            hn, cn, aw, a_ctx, p_copy = self._decode_step(
                nsteps, prev_words, hs, cs,
                cur_eh, cur_th, cur_el, cur_tl, ne,
            )

            v_scores = torch.zeros(ne, self.len_voc, device=inputs.device)
            v_scores.scatter_add_(1, cur_inp, aw)
            v_scores = torch.log(v_scores + 1e-20)

            pred = torch.log_softmax(
                self.out_dense_2(self.out_dense_1(
                    torch.cat([hn.squeeze(0), a_ctx], 1))),
                dim=-1,
            )
            fp = p_copy.unsqueeze(1) * v_scores + (1 - p_copy.unsqueeze(1)) * pred

            for b_idx in beam_dict:
                cands = []
                if ne == len(beam_dict):
                    ex_hn = hn[:, b_idx, :].unsqueeze(0)
                    ex_cn = cn[:, b_idx, :].unsqueeze(0)
                    ep = fp[b_idx]
                    _, ti = torch.sort(-ep)
                    for z in range(beam_size):
                        wi = ti[z].item()
                        cands.append((ep[wi].item(), ex_hn, ex_cn, [wi]))
                    beam_dict[b_idx] = cands
                else:
                    obs = beam_dict[b_idx]
                    s = b_idx * beam_size
                    e = s + beam_size
                    ex_hn = hn[:, s:e, :]
                    ex_cn = cn[:, s:e, :]
                    ex_fp = fp[s:e]
                    for oi, ob in enumerate(obs):
                        cp, _, _, seq = ob
                        if seq[-1] == eos_idx:
                            cands.append(ob)
                        op = ex_fp[oi]
                        oh = ex_hn[:, oi, :].unsqueeze(0)
                        oc = ex_cn[:, oi, :].unsqueeze(0)
                        _, ti = torch.sort(-op)
                        for z in range(beam_size):
                            wi = ti[z].item()
                            cands.append((
                                cp + float(op[wi].cpu().item()),
                                oh, oc, seq + [wi],
                            ))
                    si = np.argsort([x[0] for x in cands])[::-1]
                    beam_dict[b_idx] = [cands[x] for x in si][:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict


# ── Resource Download & Caching ────────────────────────────────────────

def _download_file(url, dest):
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _download_gdrive(file_id, dest):
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to download SCPN pretrained weights. "
            "Install with:  pip install gdown"
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading SCPN pretrained weights from Google Drive …")
    gdown.download(url, str(dest), quiet=False)


def _ensure_resources():
    """Download and cache all SCPN resources (models + BPE data)."""
    cache = _CACHE_DIR
    models_dir = cache / "models"
    data_dir = cache / "data"
    for d in (cache, models_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    scpn_pt = models_dir / "scpn.pt"
    pgen_pt = models_dir / "parse_generator.pt"
    vocab_pkl = data_dir / "parse_vocab.pkl"
    bpe_codes = data_dir / "bpe.codes"
    bpe_vocab = data_dir / "vocab.txt"

    if not scpn_pt.exists() or not pgen_pt.exists():
        archive = cache / "scpn_models_download"
        if not archive.exists():
            _download_gdrive(_GDRIVE_MODELS_ID, archive)

        extracted = False
        try:
            if tarfile.is_tarfile(str(archive)):
                with tarfile.open(str(archive)) as tf:
                    tf.extractall(path=str(cache), filter="data")
                extracted = True
        except Exception:
            pass
        if not extracted:
            try:
                if zipfile.is_zipfile(str(archive)):
                    with zipfile.ZipFile(str(archive)) as zf:
                        zf.extractall(path=str(cache))
                    extracted = True
            except Exception:
                pass

        for sub in cache.iterdir():
            if sub.is_dir() and sub.name not in ("models", "data"):
                for f in sub.rglob("*.pt"):
                    dest = models_dir / f.name
                    if not dest.exists():
                        f.rename(dest)

        if not extracted and not scpn_pt.exists():
            archive.rename(scpn_pt)

    for fname in ("parse_vocab.pkl", "bpe.codes", "vocab.txt"):
        dest = data_dir / fname
        if not dest.exists():
            try:
                _download_file(f"{_GITHUB_RAW}/data/{fname}", str(dest))
            except RuntimeError:
                logger.warning("Could not download %s from GitHub", fname)

    missing = []
    for path, desc in [
        (scpn_pt, "scpn.pt (SCPN encoder-decoder weights)"),
        (pgen_pt, "parse_generator.pt (ParseNet weights)"),
        (vocab_pkl, "parse_vocab.pkl (BPE word vocabulary)"),
        (bpe_codes, "bpe.codes (BPE merge operations)"),
        (bpe_vocab, "vocab.txt (BPE vocabulary)"),
    ]:
        if not path.exists():
            missing.append(f"  - {desc}: {path}")
    if missing:
        raise RuntimeError(
            "SCPN: missing required files. Download manually from\n"
            "  https://github.com/miyyer/scpn\n"
            "  https://drive.google.com/file/d/"
            f"{_GDRIVE_MODELS_ID}/view\n"
            f"and place them in {cache}:\n" + "\n".join(missing)
        )
    return {
        "scpn_pt": scpn_pt, "pgen_pt": pgen_pt,
        "vocab_pkl": vocab_pkl, "bpe_codes": bpe_codes,
        "bpe_vocab": bpe_vocab,
    }


# ── Lazy-Loaded Singletons ────────────────────────────────────────────

_scpn_net: Optional[SCPNModel] = None
_parse_net: Optional[ParseNetModel] = None
_pp_vocab: Optional[dict] = None
_rev_pp_vocab: Optional[dict] = None
_bpe = None
_parser = None


def _load_models():
    global _scpn_net, _parse_net, _pp_vocab, _rev_pp_vocab, _bpe
    if _scpn_net is not None:
        return

    paths = _ensure_resources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(paths["vocab_pkl"], "rb") as f:
        _pp_vocab, _rev_pp_vocab = pickle.load(f, encoding="latin1")

    pp_ckpt = torch.load(
        str(paths["scpn_pt"]), map_location="cpu",
        weights_only=False, encoding="latin1",
    )
    pp_args = pp_ckpt["config_args"]
    _scpn_net = SCPNModel(
        pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans,
        len(_pp_vocab), len(PARSE_GEN_VOC) - 1, pp_args.use_input_parse,
    )
    _scpn_net.load_state_dict(pp_ckpt["state_dict"])
    _scpn_net.to(device)
    _scpn_net.eval()
    logger.info(
        "SCPN model loaded (d_word=%d, d_hid=%d, d_nt=%d, d_trans=%d)",
        pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans,
    )

    pg_ckpt = torch.load(
        str(paths["pgen_pt"]), map_location="cpu",
        weights_only=False, encoding="latin1",
    )
    pg_args = pg_ckpt["config_args"]
    _parse_net = ParseNetModel(pg_args.d_nt, pg_args.d_hid, len(PARSE_GEN_VOC))
    _parse_net.load_state_dict(pg_ckpt["state_dict"])
    _parse_net.to(device)
    _parse_net.eval()
    logger.info(
        "ParseNet model loaded (d_nt=%d, d_hid=%d)", pg_args.d_nt, pg_args.d_hid,
    )

    from subword_nmt.apply_bpe import BPE, read_vocabulary
    with open(paths["bpe_codes"], encoding="utf-8") as cf:
        with open(paths["bpe_vocab"], encoding="utf-8") as vf:
            bpe_voc = read_vocabulary(vf, 50)
            cf.seek(0)
            _bpe = BPE(cf, "@@", bpe_voc, None)
    logger.info("BPE segmenter loaded")


def _load_parser():
    global _parser
    if _parser is not None:
        return _parser
    try:
        import benepar, nltk
        for res in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(res.split("/")[-1], quiet=True)
        try:
            _parser = benepar.Parser("benepar_en3")
        except LookupError:
            benepar.download("benepar_en3")
            _parser = benepar.Parser("benepar_en3")
        return _parser
    except ImportError:
        raise RuntimeError(
            "benepar is required for SCPN constituency parsing. "
            "Install: pip install benepar && "
            "python -c \"import benepar; benepar.download('benepar_en3')\""
        )


# ── Main Attack Pipeline ──────────────────────────────────────────────

def run_scpn(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_templates: int = 10,
    similarity_threshold: float = 0.7,
    beam_size: int = 3,
) -> str:
    """SCPN adversarial attack — exact match to Iyyer et al. (2018).

    Runs the full two-stage pipeline from generate_paraphrases.py:
    ParseNet (template → full parse) → SCPN (sentence + parse → paraphrase).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "SCPN: starting (templates=%d, sim=%.2f, beam=%d)",
        num_templates, similarity_threshold, beam_size,
    )

    _load_models()
    device = next(_scpn_net.parameters()).device

    # ── Step 1: original prediction ────────────────────────────────
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # ── Step 2: constituency-parse → linearize ─────────────────────
    parser = _load_parser()
    tree = parser.parse(text)
    parse_tokens = _deleaf(tree)
    logger.info("SCPN: source parse (%d tokens)", len(parse_tokens))

    oov = [t for t in parse_tokens if t not in PARSE_GEN_VOC]
    if oov:
        logger.warning("SCPN: OOV parse tokens (skipped): %s", oov)
        parse_tokens = [t for t in parse_tokens if t in PARSE_GEN_VOC]
    if not parse_tokens:
        logger.warning("SCPN: empty parse after OOV removal")
        return text

    np_parse = np.array(
        [PARSE_GEN_VOC[t] for t in parse_tokens], dtype="int32",
    )
    t_parse = torch.from_numpy(np_parse).long().to(device)
    t_parse_len = torch.tensor([len(parse_tokens)], dtype=torch.long, device=device)

    # ── Step 3: encode target templates ────────────────────────────
    templates = SCPN_TEMPLATES[:num_templates]
    tmpl_lens = [len(t.split()) for t in templates]
    np_tmpls = np.zeros((len(templates), max(tmpl_lens)), dtype="int32")
    for z, tmpl in enumerate(templates):
        toks = tmpl.split()
        np_tmpls[z, : len(toks)] = [PARSE_GEN_VOC[w] for w in toks]
    t_tmpls = torch.from_numpy(np_tmpls).long().to(device)
    t_tmpl_lens = torch.tensor(tmpl_lens, dtype=torch.long, device=device)

    # ── Step 4: ParseNet — templates → full parses ─────────────────
    pg_beams = _parse_net.batch_beam_search(
        t_parse.unsqueeze(0), t_tmpls,
        t_parse_len, t_tmpl_lens,
        PARSE_GEN_VOC["EOP"],
        beam_size=beam_size, max_steps=150,
    )

    seqs, seq_lens = [], []
    for b_idx in pg_beams:
        _, _, _, seq = pg_beams[b_idx][0]
        seq = seq[:-1] if seq and seq[-1] == PARSE_GEN_VOC["EOP"] else seq
        if seq:
            seqs.append(seq)
            seq_lens.append(len(seq))

    if not seqs:
        logger.info("SCPN: ParseNet produced no valid parses")
        return text

    np_parses = np.zeros((len(seqs), max(seq_lens)), dtype="int32")
    for z, seq in enumerate(seqs):
        np_parses[z, : seq_lens[z]] = seq
    t_parses = torch.from_numpy(np_parses).long().to(device)
    t_plens = torch.tensor(seq_lens, dtype=torch.long, device=device)

    # ── Step 5: BPE-segment input sentence ─────────────────────────
    seg = _bpe.segment(text.lower()).split()
    seg_ids = [_pp_vocab[w] for w in seg if w in _pp_vocab]
    seg_ids.append(_pp_vocab["EOS"])
    t_sent = torch.tensor(seg_ids, dtype=torch.long, device=device)
    t_sent_len = torch.tensor([len(seg_ids)], dtype=torch.long, device=device)

    # ── Step 6: SCPN — sentence + parses → paraphrases ────────────
    pp_beams = _scpn_net.batch_beam_search(
        t_sent.unsqueeze(0), t_parses,
        t_sent_len, t_plens,
        _pp_vocab["EOS"],
        beam_size=beam_size, max_steps=40,
    )

    # ── Step 7: decode, filter, evaluate ───────────────────────────
    candidates = []
    for b_idx in pp_beams:
        _, _, _, seq = pp_beams[b_idx][0]
        words = [_rev_pp_vocab.get(w, "") for w in seq[:-1]]
        gen_text = _reverse_bpe(words)
        if not gen_text.strip() or gen_text.lower() == text.lower():
            continue

        overlap = _ngram_overlap(text, gen_text)
        if overlap < 0.5:
            continue

        sim = compute_semantic_similarity(text, gen_text)
        if sim < similarity_threshold:
            continue

        candidates.append({"text": gen_text, "sim": sim, "tmpl": b_idx})

    if not candidates:
        logger.info("SCPN: no candidates passed filters")
        return text

    candidates.sort(key=lambda c: c["sim"], reverse=True)

    # ── Step 8: adversarial selection ──────────────────────────────
    best_text = text
    best_impact = 0.0
    evaluated = 0

    for c in candidates:
        evaluated += 1
        label, conf, _ = model_wrapper.predict(c["text"])

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info(
                    "SCPN: success at candidate %d (tmpl=%d, sim=%.3f)",
                    evaluated, c["tmpl"], c["sim"],
                )
                return c["text"]
        else:
            if label != orig_label:
                logger.info(
                    "SCPN: success at candidate %d (tmpl=%d, sim=%.3f)",
                    evaluated, c["tmpl"], c["sim"],
                )
                return c["text"]

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = c["text"]

    logger.info("SCPN: done (%d candidates evaluated)", evaluated)
    return best_text

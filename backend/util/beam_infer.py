import os, sys
import torch
from dataclasses import dataclass
from typing import List
import torch.nn.functional as F
import torchaudio, math
from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder, CTCDecoder

from flashlight.lib.text.dictionary import (
    create_word_dict as _create_word_dict,
    Dictionary as _Dictionary,
    load_words as _load_words,
)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
        List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return indices


class TreeNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None

def build_tree(lexicon):
    root = TreeNode()
    for word, chars in lexicon.items():
        node = root
        for c in chars:
            if c not in node.children:
                node.children[c] = TreeNode()
            node = node.children[c]
        node.is_word = True
        node.word = word
    return root
                    
def print_trie(node, prefix=""):
    for char, child in node.children.items():
        if child.is_word:
            print(prefix + char + "  →  " + child.word)
        else:
            print(prefix + char)
        print_trie(child, prefix + "  ")
        
def print_node(node, label):
    print(label, end=": ")
    print(node.word)
    
class StreamingGreedyLexiconCTC:
    def __init__(self, trie_root, tokens=None, blank=0):
        self.trie_root = trie_root
        self.blank = tokens[blank]
        self.tokens=tokens
        self.reset()
    def reset(self):
        self.node = self.trie_root
        self.prev_token = None
        self.current_tokens = []
        self.candidate_word = None   # parola trovata ma NON ancora emessa
        self.words = []

    def _commit_word(self):
        if self.candidate_word is not None:
            self.words.append(self.candidate_word)
        # reset stato parola
        self.candidate_word = None
        self.current_tokens = []
        self.node = self.trie_root

    def step(self, frame_logits):
        token = self.tokens[frame_logits.argmax().item()]
        # Regole CTC: blank = fine parola
        if token == self.blank:
            #self._commit_word()
            #self.prev_token = token
            return

        # Regola CTC: ignora ripetizioni
        if token == self.prev_token:
            self.prev_token = token
            return

        self.prev_token = token

        # Token valido nel trie?
        if token in self.node.children:
            #print_node(self.node,"FATHER")
            self.node = self.node.children[token]
            #print_node(self.node,"CHILDREN")            
            self.current_tokens.append(token)

            # Se e` una parola completa si segna come candidata (ma NON emettere)
            if self.node.is_word:
                #print("WORD: ",self.node.word)
                self.candidate_word = self.node.word
        else:
            # Token non valido e` commit della parola candidata
            self._commit_word()
            #print_node(self.node,"COMMIT")
            # Prova a ripartire da zero con questo token
            if token in self.trie_root.children:
                self.node = self.trie_root.children[token]
                #print_node(self.node,"RIPARTITO")
                self.current_tokens = [token]
                if self.node.is_word:
                    self.candidate_word = self.node.word
            else:
                # Token completamente fuori lessico
                self.node = self.trie_root
                self.current_tokens = []
                self.candidate_word = None

    def get_partial(self):
        return list(self.words)

    def finalize(self):
        self._commit_word()
        return list(self.words)


def logsumexp(a, b):
    if a == -float('inf'): return b
    if b == -float('inf'): return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))

class BeamState:
    def __init__(self, node, tokens, cand_word, words,
                 p_b, p_nb, prev_token):
        self.node = node
        self.tokens = tokens
        self.cand_word = cand_word
        self.words = words
        self.p_b = p_b
        self.p_nb = p_nb
        self.prev_token = prev_token

    @property
    def logp(self):
        return logsumexp(self.p_b, self.p_nb)

    def key(self):
        return (id(self.node), tuple(self.tokens), tuple(self.words), self.cand_word)

class StreamingCTCBeamSearchLexicon:
    def __init__(self, trie_root, token_dict, blank, beam_size=5):
        self.trie_root = trie_root
        self.token_dict = token_dict
        self.blank = blank
        self.beam_size = beam_size
        self.reset()

    def reset(self):
        self.beams = [
            BeamState(
                node=self.trie_root,
                tokens=[],
                cand_word=None,
                words=[],
                p_b=0.0,
                p_nb=-float('inf'),
                prev_token=None
            )
        ]

    def step(self, frame_logprobs):
        new_beams = {}

        for beam in self.beams:
            V = frame_logprobs.size(0)

            for tok_ in range(V):
                token = self.token_dict[tok_]
                lp = frame_logprobs[tok_].item()

                # --- CTC prefix update ---
                if tok_ == self.blank:
                    new_p_b = logsumexp(beam.p_b, beam.p_nb) + lp
                    new_p_nb = -float('inf')
                    new_prev = tok_
                    node = beam.node
                    tokens = list(beam.tokens)
                    cand_word = beam.cand_word
                    words = list(beam.words)

                else:
                    if tok_ == beam.prev_token:
                        new_p_nb = beam.p_nb + lp
                    else:
                        new_p_nb = logsumexp(beam.p_b, beam.p_nb) + lp

                    new_p_b = -float('inf')
                    new_prev = tok_

                    # --- Trie integration ---
                    node = beam.node
                    tokens = list(beam.tokens)
                    cand_word = beam.cand_word
                    words = list(beam.words)

                    if token in node.children:
                        node = node.children[token]
                        tokens.append(token)
                        if node.is_word:
                            cand_word = node.word
                    else:
                        if cand_word is not None:
                            words = words + [cand_word]
                        cand_word = None
                        tokens = []
                        node = self.trie_root

                        if token in node.children:
                            node = node.children[token]
                            tokens = [token]
                            if node.is_word:
                                cand_word = node.word
                        else:
                            continue  # percorso OOV → scartato

                new_state = BeamState(node, tokens, cand_word, words,
                                      new_p_b, new_p_nb, new_prev)

                k = new_state.key()
                if k not in new_beams:
                    new_beams[k] = new_state
                else:
                    old = new_beams[k]
                    old.p_b = logsumexp(old.p_b, new_state.p_b)
                    old.p_nb = logsumexp(old.p_nb, new_state.p_nb)

        beams = list(new_beams.values())
        beams.sort(key=lambda b: b.logp, reverse=True)
        self.beams = beams[:self.beam_size]

    def get_partial(self):
        best = max(self.beams, key=lambda b: b.logp)
        out = list(best.words)
        if best.cand_word is not None:
            out.append(best.cand_word)
        return out

    def finalize(self):
        for b in self.beams:
            if b.cand_word is not None:
                b.words = b.words + [b.cand_word]
                b.cand_word = None
        best = max(self.beams, key=lambda b: b.logp)
        return list(best.words)

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


class BeamInference(object):

    def __init__(self, args):
        self.args = args

        # for bigger LM
        self.LM_WEIGHT = 1.0  # 3.23#1.0#3.23
        self.WORD_SCORE = -4  # -1.0#-0.26
        self.N_BEST = 1  # 500#300

        '''
        #for smaller LM
        self.LM_WEIGHT = 10.0
        self.WORD_SCORE = -0.26
        self.N_BEST = 1
        '''

        if args.bpe == True:
            lex_dict = {}
            tok_dict = {}
            with open(args.lexicon, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    parola = parts[0]
                    caratteri = parts[1:]
                    lex_dict[parola] = caratteri

            with open(args.tokens, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    tok_dict[i] = line.strip()

            tree_lex = build_tree(lex_dict)
            #print_trie(tree_lex)
            #sys.exit()
            self.decoder = []
            # for w_ins in [-1,-1,-1,-1,-1, -1]: #valori positivi aumentano le inserzioni
            for w_ins in [0, 0, 0, 0, 0, 0]:  # valori positivi aumentano le inserzioni
                # for w_ins in [-0.5,0.5,0.5,0.5,0.5, 0.5]:
                self.decoder += [ctc_decoder(lexicon=args.lexicon,
                                             tokens=args.tokens,
                                             nbest=self.N_BEST,
                                             log_add=False,
                                             beam_size=args.beam_size,  # 0, #500,
                                             word_score=w_ins,
                                             lm_weight=self.LM_WEIGHT,
                                             blank_token="@",
                                             unk_word="<unk>",
                                             sil_token="<pad>")]
            self.s_decoder_ = ctc_decoder(lexicon=args.lexicon,
                                        tokens=args.tokens,
                                        nbest=self.N_BEST,
                                        log_add=False,
                                        beam_size=args.beam_size,  # 0, #500,
                                        word_score=0,
                                        lm_weight=self.LM_WEIGHT,
                                        blank_token="@",
                                        unk_word="<unk>",
                                        sil_token="<pad>")
            #self.s_decoder=StreamingGreedyLexiconCTC(tree_lex, tokens=tok_dict, blank=0)
            self.s_decoder=StreamingCTCBeamSearchLexicon(tree_lex, token_dict=tok_dict, blank=0)
        else:
            self.beam_search_decoder = ctc_decoder(
                lexicon=args.lexicon,
                tokens=args.tokens,
                nbest=1,
                log_add=True,
                beam_size=args.beam_size,
                lm_weight=self.LM_WEIGHT,
                word_score=self.WORD_SCORE
            )

        # lm="lm.bin"
        # lm="4gram_small.arpa.lm"
        if args.device == "cuda":
            self.cuda_decoder = cuda_ctc_decoder(
                args.tokens, nbest=1, beam_size=args.beam_size, blank_skip_threshold=0.95)

        self.greedy_decoder = GreedyCTCDecoder()

    def stream_decoder(self,emission=None, partial=False):
        if partial:
            emission=emission.squeeze(0)
            for t in range(emission.size(0)):
                self.s_decoder.step(emission[t])
            partial_transc = self.s_decoder.get_partial()
            return(partial_transc)
        else:
            final = self.s_decoder.finalize()
            self.s_decoder.reset()
        return(final)
            
    def beam_predict(self, model, input_sequence):
        emission = model.ctc_encoder(input_sequence)
        beam_search_result = self.beam_search_decoder(emission.cpu())
        beam_search_transcript = " ".join(
            beam_search_result[0][0].words).strip()
        return (beam_search_transcript)


    def ctc_predict_(self, emission, index=5):
        beam_search_result = self.decoder[index](emission.cpu())
        beam_search_transcript = []
        for s_ in beam_search_result:
            beam_search_transcript = beam_search_transcript + \
                [" ".join(s_[0].words).strip()]
        return (beam_search_transcript)


    def ctc_cuda_predict(self, emission, tokens=None):
        if tokens == None:
            tokens = self.args.tokens
        
        enc_len = torch.full(size=(emission.size(0),), fill_value=emission.size(
            1), dtype=torch.int32).to(self.args.device)
        cuda_decoder = cuda_ctc_decoder(
            tokens, nbest=1, beam_size=self.args.beam_size, blank_skip_threshold=0.95)
        
        results = cuda_decoder(emission, enc_len)
        return (results)
    

    def ctc_predict(self, emission, index=5):
        beam_search_result = self.decoder[index](emission.cpu())
        beam_search_transcript = [
            " ".join(beam_search_result[0][0].words).strip()]
        nhyps = len(beam_search_result[0])
        hyp_score = torch.zeros(nhyps)

        for i in range(0, nhyps):
            hyp_score[i] = beam_search_result[0][i].score

        pprob = F.softmax(hyp_score, dim=0)
        return beam_search_transcript, pprob[0]


    def get_trellis(self, emission, tokens, blank_id=0):

        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(self.args.device)
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis


    def backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        # t_start = torch.argmax(trellis[:, j]).item()
        t_start = trellis.size(0)-1
        path = []
        prob = 0
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            # prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            prob = prob + emission[t - 1, tokens[j - 1]
                                   if changed > stayed else 0].item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        if j > 0:
            # raise ValueError("Failed to align")
            print(t, j, "Failed to align")
        return path[::-1]


    def sequence_length_penalty(self, length: int, alpha: float = 0.6) -> float:
        return ((5 + length) / (5 + 1)) ** alpha


    def beam_search(self, model, encoder_output, layer_n, 
                    vocab_size=None, max_length=500, min_length=300, 
                    SOS_token=None, EOS_token=None, PAD_token=None, 
                    beam_size=None, pen_alpha=None, return_best_beam=True):

        if vocab_size == None:
            vocab_size = self.args.dec_voc_size
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx
        if beam_size == None:
            beam_size == self.args.beam_size
        if pen_alpha == None:
            pen_alpha = self.args.pen_alpha
            
        beam_size_count = beam_size
        
        # decoder_input = input_decoder[:,0:input_decoder.size(1)-5]
        decoder_input = torch.tensor(
            [[SOS_token]], dtype=torch.long, device=self.args.device)

        scores = torch.Tensor([0.]).to(self.args.device)
        # print("DECODER_INPUT:",  text_transform.int_to_text(decoder_input.squeeze(0)))
        # input_sequence = input_sequence.to(device)

        # encoder_output = model._encoder_(input_sequence, valid_length, layer_n).to(device)

        # _,emission = model(input_sequence,decoder_input)
        final_scores = []
        final_tokens = []

        for i in range(max_length):
            # decoder_input = F.pad(decoder_input, (0,10), mode='constant',value=PAD_token)

            if i == 0:
                logits = model._decoder_(
                    decoder_input, encoder_output, layer_n).detach()
            else:
                logits = model._decoder_(decoder_input, encoder_output.expand(
                    beam_size_count, *encoder_output.shape[1:]), layer_n).detach()

            log_probs = logits[:, -1] / self.sequence_length_penalty(i+1, pen_alpha)
            scores = scores.unsqueeze(1) + log_probs
            scores, indices = torch.topk(scores.reshape(-1), beam_size_count)

            beam_indices = torch.divide(
                indices, vocab_size, rounding_mode='floor')
            token_indices = torch.remainder(indices, vocab_size)
            next_decoder_input = []
            EOS_beams_index = []

            for ind, (beam_index, token_index) in enumerate(zip(beam_indices, token_indices)):

                prev_decoder_input = decoder_input[beam_index]

                if token_index == EOS_token and i > min_length:

                    token_index = torch.LongTensor([token_index]).to(self.args.device)
                    final_tokens.append(
                        torch.cat([prev_decoder_input, token_index]))
                    final_scores.append(scores[ind])
                    beam_size_count -= 1

                    # scores_list = scores.tolist()
                    # del scores_list[ind]
                    # scores = torch.tensor(scores_list, device=device)
                    EOS_beams_index.append(ind)
                    # print(f"Beam #{ind} reached EOS!")

                else:
                    token_index = torch.LongTensor(
                        [token_index]).to(self.args.device)
                    next_decoder_input.append(
                        torch.cat([prev_decoder_input, token_index]))
            if len(EOS_beams_index) > 0:
                scores_list = scores.tolist()
                for tt in EOS_beams_index[::-1]:
                    del scores_list[tt]
                scores = torch.tensor(scores_list, device=self.args.device)

            if len(final_scores) == beam_size:
                break

            decoder_input = torch.vstack(next_decoder_input)

        # We have reached max # of allowed iterations.
        if i == (max_length - 1):

            for beam_unf, score_unf in zip(decoder_input, scores):
                final_tokens.append(beam_unf)
                final_scores.append(score_unf)
                del beam_unf
                del score_unf

            assert len(final_tokens) == beam_size and len(final_scores) == beam_size, (
                'Final_tokens and final_scores lists do not match beam_size size!')

        # If we want to return most probable predicted beam.
        # del encoder_output
        # del encoder_output_afterEOS
        del decoder_input
        del scores
        if return_best_beam:
            del encoder_output
            max_val = max(final_scores)

        return final_tokens, final_scores, final_tokens[final_scores.index(max_val)].tolist()
        
        # else:

        #     s_ctc = torch.zeros(beam_size)
        #     # loss_ctc = torch.zeros(beam_size)
        #     i = 0
        #     ctc_input_len = torch.full(
        #         size=(emission.size(0),), fill_value=emission.size(1), dtype=torch.long)

        #     # for f_t, f_s in zip(final_tokens, final_scores):
        #     for f_t in final_tokens:
        #         # f_t=f_t[1:f_t.size(0)-1]
        #         # print(f_t)

        #         trellis = self.get_trellis(
        #             emission.squeeze(0).to(self.args.device), f_t).detach()
        #         path = self.backtrack(trellis, emission.squeeze(0), f_t)
        #         # print(path[0].score/len(path), len(path))
        #         '''
        #         stayed=path[0]
                
        #         count = 0
        #         s_ctc[i] = 0
        #         for p in path:
        #             #print(p.score, p.token_index, stayed.token_index)
        #             if p.token_index != stayed.token_index:
        #                 s_ctc[i] = s_ctc[i] + ( (stayed.score - pc.score) / count)
        #                 #print("stayed", pc.score, stayed.score, count, s_ctc[i])
        #                 stayed=p
        #                 count = 1
                        
        #             else:
        #                 count = count + 1
        #             pc=p
        #         s_ctc[i] = s_ctc[i] + ( (stayed.score - pc.score) / count)
        #         '''
        #         # print("final", pc.score, stayed.score, count, s_ctc[i])
        #         # plt.imshow(trellis[1:, 1:].T, origin="lower")
        #         # plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
        #         # plt.colorbar()
        #         # plt.show()

        #         ctc_target_len = f_t.size(0)
        #         # s_ctc[i] = ctc_loss(emission.permute(1,0,2),f_t.unsqueeze(0),ctc_input_len,torch.tensor(ctc_target_len)).to(device)#/len(f_t)
        #         s_ctc[i] = path[0].score/len(f_t)  # len(f_t)#len(path)
        #         i = i+1

        #     s_pred = torch.exp(torch.tensor(final_scores))
        #     s_ctc = torch.exp(s_ctc)
        #     # print("PRED:",s_pred)
        #     # print("CTC:",s_ctc)

        #     # s_pred = s_pred / torch.sum(s_pred)
        #     # s_ctc = s_ctc / torch.sum(s_ctc)

        #     # loss_ctc=torch.exp(loss_ctc/len(path))

        #     max_ = torch.max(s_pred, dim=0, keepdim=False)
        #     s_pred = s_pred / max_.values
        #     max_ = torch.max(s_ctc, dim=0, keepdim=False)
        #     s_ctc = s_ctc / max_.values

        #     # print("PRED:",s_pred)
        #     # print("CTC:",s_ctc)

        #     # print("LOSS:",loss_ctc)
        #     s_norm = s_ctc * weight_ctc + s_pred * \
        #         (1-weight_ctc)  # + 0.5 * s_lm

        #     # min_=torch.min(s_norm,dim=0,keepdim=False)
        #     max_ = torch.max(s_norm, dim=0, keepdim=False)
        #     # max_val = max(s_norm)
        #     del encoder_output
        #     # del trellis
        #     # del path
        #     return final_tokens, final_scores, final_tokens[max_.indices].tolist()

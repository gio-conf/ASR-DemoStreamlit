from util.conf import get_args
from inference_online import Wrapper
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference

class Server():
    def __init__(self):    
        self.args = get_args()
        self.model = Early_conformer(src_pad_idx=self.args.src_pad_idx,
                                    n_enc_exits=self.args.n_enc_exits,
                                    d_model=self.args.d_model,
                                    enc_voc_size=self.args.enc_voc_size,
                                    dec_voc_size=self.args.dec_voc_size,
                                    max_len=self.args.max_len,
                                    d_feed_forward=self.args.d_feed_forward,
                                    n_head=self.args.n_heads,
                                    n_enc_layers=self.args.n_enc_layers_per_exit,
                                    features_length=self.args.n_mels,
                                    drop_prob=self.args.drop_prob,
                                    depthwise_kernel_size=self.args.depthwise_kernel_size,
                                    device=self.args.device).to(self.args.device)
        self.inf = BeamInference(args=self.args)

        self.w = Wrapper(args=self.args, model=self.model, inf=self.inf)
    
    def start_server(self):
        print("Server running")
        self.w.run(self.args, self.model, self.inf)
        while True:
            print(self.w.transcription)

s = Server()
s.start_server()
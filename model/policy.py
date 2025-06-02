# model/policy.py

from model.cnn    import *
from model.memory import *

class cnnrnnattn_policy(nn.Module):
    '''
    input : state, hx
    model : CNN → RNN → ATTENTION
    output: logits, value, state_emd, hidden_emd, chosen_ids, α, full-W
    '''
    def __init__(self, params):
        super().__init__()
        self.cnn_embed_lyr   = cnn_embed(**params['cnn_embed'])
        self.rnn_lyr         = nn.RNN(**params['rnn'])
        self.memory_gate_lyr = memory_gate(**params['memory_gate'])

    def forward(self, state, hx, memory_bank_set):
        # 1) 현재 obs 를 CNN으로 임베드 → query_state
        state_emd   = self.cnn_embed_lyr(state).unsqueeze(0)   # 1×1×hidden_dim

        # 2) RNN 으로 은닉 벡터 계산 → hidden_state
        hidden_emd, _ = self.rnn_lyr(state_emd, hx)            # 1×1×hidden_dim

        # 3) Memory Gate 에는 총 7개의 인자를 넘겨야 함
        logits, value, attention_weights, chosen_ids, gate_alpha, attention_weights_full = \
            self.memory_gate_lyr(
                state_emd,                        # query_state  : CNN 임베딩
                hidden_emd,                       # hidden_state : RNN 은닉
                memory_bank_set.embed_memory,     # embed_memory : CNN-embed 캐시  ★MOD
                memory_bank_set.hidden_memory,    # hidden_memory: RNN-은닉 캐시
                memory_bank_set.action_memory,    # action_memory: 슬롯별 action
                memory_bank_set.reward_memory,    # reward_memory: 슬롯별 reward
                memory_bank_set.memory_ids        # memory_ids   : 슬롯 ID
            )

        return logits, value, state_emd, hidden_emd, chosen_ids, gate_alpha, attention_weights_full
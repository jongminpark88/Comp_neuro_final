from model.cnn import * 
from model.memory import *

class cnnrnnattn_policy(nn.Module):
    '''
    input: state, hx
    model: CNN > RNN > ATTENTION
    output: log_probs, new_state, state_emd, attention weights
    '''
    def __init__(self, params):
        super().__init__()
        self.cnn_embed_lyr = cnn_embed(**params['cnn_embed'])
        self.rnn_lyr = nn.RNN(**params['rnn']) 
        self.memory_gate_lyr = memory_gate(**params['memory_gate'])
    
    def forward(self, state, hx, memory_bank_set):
        state_emd = self.cnn_embed_lyr(state).unsqueeze(0)
        hidden_emd, _ = self.rnn_lyr(state_emd, hx)
        log_probs, value, attention_weights, chosen_ids, gate_alpha, attention_weights_full = self.memory_gate_lyr(hidden_emd, 
                                                                                    memory_bank_set.hidden_memory,
                                                                                    memory_bank_set.action_memory,
                                                                                    memory_bank_set.reward_memory,
                                                                                    memory_bank_set.memory_ids,
                                                                                    )
        return log_probs, value, state_emd, hidden_emd, chosen_ids, gate_alpha, attention_weights_full


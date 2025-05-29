from model.utils import * 

class memory_bank():
    def __init__(
            self,
            decay_rate: float=0.0001, 
            noise_std: float=0.001,
            et_lambda: float=0.99,
            memory_len: int=5000,
            update_freq: int=100,
            hidden_dim: int=128,
            decay_yn: bool=False
    ):
        self.decay_rate = decay_rate
        self.noise_std = noise_std
        self.et_lambda = et_lambda 
        self.update_freq = update_freq
        self.memory_len = memory_len
        self.decay_yn = decay_yn

        self.memory_id = 0 ## 시간 순서대로의 memory_id
        self.memory_bank_org = {}
        self.hidden_memory = torch.empty(1, 1, 128)
        self.action_memory = torch.empty(0, dtype=torch.int)
        self.reward_memory = torch.empty(0)
        
        self.memory_ids = torch.empty(0, dtype=torch.int)
        self.trace = torch.empty(0)

    def update(self, retina, embed_state, hidden_state, action, reward, obs, ep, chosen_ids):

        ## 1) push & truncate
        self.push(retina, embed_state, hidden_state, action, reward, obs, ep) ## push (5/28 확인완료)
        self.update_trace(chosen_ids) ## update trace (5/28 확인완료)
        self.trunc() ## truncate (5/28 확인완료)

        ## 2) hard coded everything
        self.hidden_memory = torch.cat([element[3] for element in self.memory_bank_org.values()], dim=1)
        self.action_memory = torch.tensor([element[4] for element in self.memory_bank_org.values()], dtype=torch.int)
        self.reward_memory = torch.Tensor([element[5] for element in self.memory_bank_org.values()])
        # self.hidden_memory = torch.cat(list(np.array(list(self.memory_bank_org.values()))[:,3]), dim=2) ## memory_slot에서 3번째에 해당하는 걸 dim=2 기준으로 concat
        # self.action_memory = torch.tensor(list(np.array(list(self.memory_bank_org.values()))[:,4]), dtype=torch.int)
        # self.reward_memory = torch.tensor(list(np.array(list(self.memory_bank_org.values()))[:,5]))

    def push(self, retina, embed_state, hidden_state, action, reward, obs, ep):
        
        if self.decay_yn:
            hidden_state_decay = self.add_decay(hidden_state)
        else:
            hidden_state_decay = hidden_state

        self.memory_id += 1 ## memory_id 1에서 시작
        
        ## ids 업데이트
        self.memory_ids = torch.cat((self.memory_ids, torch.tensor([self.memory_id], dtype=torch.int)))

        ## memory slot 정의
        self.memory_slot = [
            retina,
            embed_state.detach().clone(), # tensor
            hidden_state.detach().clone(), # tensor
            hidden_state_decay.detach().clone(), # tensor
            action, # int
            reward, # int
            obs['timestep'],
            obs['position'],
            ep
        ]

        ## memory bank 업데이트
        self.memory_bank_org[self.memory_id] = self.memory_slot

    def trunc(self):
        ### memory_bank에 memory_len 이상의 memory가 쌓였을 때 & update_freq에 도달했을 때
        if (self.memory_id > self.memory_len) & (self.memory_id % self.update_freq == 0):     
            ### trace, with more recently added one favored
            # random_idx = torch.rand_like(trace).argsort(dim=0)
            # trace_random = torch.take_along_dim(trace, random_idx, dim=0)
            # _, bottomk_random_idx = torch.topk(trace_random, k=self.update_freq, largest=False, dim=0)
            # bottomk_idx = torch.take_along_dim(random_idx, bottomk_random_idx, dim=0)
            bottomk_idx = gen_topk_random(self.trace, k=self.update_freq, dim=0, largest=False)
            
            ## memory_bank_org에서 삭제 (5010 > 5000)
            for each_idx in bottomk_idx: 
                each_id = self.memory_ids[each_idx]
                del self.memory_bank_org[each_id]

            ## ids, trace에서 삭제 (5010 > 5000)
            mask = torch.ones(self.trace.size(0), dtype=torch.bool)
            mask[bottomk_idx] = False

            self.memory_ids = self.memory_ids[mask]
            self.trace = self.trace[mask]

    def update_trace(self, chosen_ids):
        self.trace *= self.et_lambda ## 과거 memory에 대한 trace decay
        self.trace = torch.cat((self.trace, torch.tensor([1]))) ## new memory에 대한 trace 추가
        if chosen_ids.nelement() != 0: #만약 memory가 없어 chosen id도 없다면 
            add_recency = torch.isin(self.memory_ids, chosen_ids).to(dtype=torch.int) ## memory_ids에서 chosen_ids가 있는 자리를 골라냄
            self.trace += add_recency

    def add_decay(self, hidden_state):
        decay_prod = math.exp(self.decay_rate * self.memory_id)
        hidden_state_decay = hidden_state + torch.normal(0, self.noise_std, size=hidden_state.shape)*decay_prod
        return hidden_state_decay

    def save(self, timestep_basedir, attention_base_dir, ep, timestep, chosen_ids):
        os.makedirs(os.path.join(timestep_basedir, f'ep{ep}'), exist_ok=True)
        os.makedirs(os.path.join(attention_base_dir, f'ep{ep}'), exist_ok=True)
        timestep_memory = {self.memory_id : self.memory_slot}
        attn_memory = {chosen.item(): self.memory_bank_org[chosen.item()] for chosen in chosen_ids}
        
        with open(os.path.join(timestep_basedir, f'ep{ep}', f'timestep_memory_{timestep}.pkl'), 'wb') as file1:
            pickle.dump(timestep_memory, file1)
        with open(os.path.join(attention_base_dir, f'ep{ep}', f'attention_memory_{timestep}.pkl'),'wb') as file2:
            pickle.dump(attn_memory, file2)

class memory_gate(nn.Module):
    """
    input: current hidden state, memory_bank_hidden
    output: current hidden state -- attention -- many memory_bank_hidden memorys > outputs action
    """
    def __init__(
            self, 
            hidden_dim_lyrs: list=[128, 64],
            action_dim: int=7,
            attn_size: int=5,
            rl_algo_arg: str='default'
        ):
        
        super().__init__()

        ### 0) ATTENTION WEIGHTS 
        self.hidden_dim = hidden_dim_lyrs[0]
        self.attn_size = attn_size
        self.action_dim = action_dim
        self.Q = torch.nn.Parameter(torch.randn(self.hidden_dim,  self.hidden_dim) / math.sqrt(self.hidden_dim)) ## 128 x 128
        self.K = torch.nn.Parameter(torch.randn(self.hidden_dim,  self.hidden_dim) / math.sqrt(self.hidden_dim)) ## 128 x 128
        self.V = torch.nn.Parameter(torch.randn(self.hidden_dim,  self.hidden_dim) / math.sqrt(self.hidden_dim)) ## 128 x 128
        self.gate_alpha = torch.nn.Parameter(torch.zeros(1))
        # self.F = torch.nn.Parameter(torch.randn(hidden_dim,  hidden_dim) / math.sqrt(hidden_dim))

        ### 1) LINEAR LAYERS
        # self.lin_lyrs = [
        #     nn.ReLU(), 
        #     nn.Linear(2*self.hidden_dim, hidden_dim_lyrs[1])
        #     ]
        self.rl_algo_arg = rl_algo_arg
        self.lin_lyrs_actor = []
        if rl_algo_arg == 'A2C':
            self.lin_lyrs_critic = []
        lin_lyr_prev = hidden_dim_lyrs[0] + 2*self.attn_size
        for lin_lyr in hidden_dim_lyrs[1:]:
            self.lin_lyrs_actor += [
                nn.ReLU(),
                nn.Linear(lin_lyr_prev, lin_lyr)
            ]
            if rl_algo_arg == 'A2C':
                self.lin_lyrs_critic += [
                    nn.ReLU(),
                    nn.Linear(lin_lyr_prev, lin_lyr)
                ]
            lin_lyr_prev = lin_lyr

        self.lin_lyrs_actor += [nn.ReLU(), nn.Linear(lin_lyr_prev, action_dim)]
        self.lin_actor = nn.Sequential(*self.lin_lyrs_actor)

        if rl_algo_arg == 'A2C':
            self.lin_lyrs_critic += [nn.ReLU(), nn.Linear(lin_lyr_prev, 1)]
            self.lin_critic = nn.Sequential(*self.lin_lyrs_critic)
            # self.softmax_lyr = nn.Softmax(dim=0)
        
    def forward(self, hidden_state, hidden_memory, action_memory, reward_memory, memory_ids):
        
        value = torch.empty(0)
        if hidden_memory.size(1) >= 5:
            ### 0) Memory_idx_lst update and QUERY, KEY, VALUE CALCULATION
            Query = torch.matmul(hidden_state, self.Q) ## 1 x 1 x 128
            Key = torch.squeeze(torch.matmul(hidden_memory, self.K)) ## 1 x 100 x 128 -- squeezed -- > 100 x 128          
            Value = torch.matmul(hidden_memory, self.V) ## 1 x 100 x 128

            ### 1) ATTENTION
            Wattn = torch.matmul(Query, Key.T) / math.sqrt(self.hidden_dim) ## 1 x 1 x 100
            Wattn = nn.Softmax(dim=2)(Wattn) ## 1 x 1 x 100

            ### added - Top 5 Attentions 
            Wattn_topk_idx = gen_topk_random(Wattn, k=self.attn_size, dim=2, largest=True)
            # Wattn_topk_val, Wattn_topk_idx = torch.topk(Wattn, k=self.attn_size, dim=2) ## 1 x 1 x 5
            Wattn_topk_val = Wattn[:, :, Wattn_topk_idx] 
            Wattn_topk_idx_ = torch.squeeze(Wattn_topk_idx)
            Value_topk_val = Value[:, Wattn_topk_idx_, :] ## 1 x 5 x 128

            chosen_attn = torch.matmul(Wattn_topk_val, Value_topk_val) ## 1 x 1 x 128
            chosen_memory_ids = torch.stack([memory_ids[i] for i in Wattn_topk_idx_])
            chosen_action_feature = action_memory[Wattn_topk_idx_]
            chosen_reward_feature = reward_memory[Wattn_topk_idx_]

            ### 2) OUTPUT ACTIONS
            # if memory_bank_len > memory_bank_hidden.size(1):
            #     new_memory = torch.cat([hidden_state, chosen_action_feature]) ## 1 x 1 x 128
            # else:
            new_memory_hidden = (1-self.gate_alpha)*hidden_state + self.gate_alpha*chosen_attn ## 1 x 1 x 128
            new_memory = torch.cat([torch.squeeze(new_memory_hidden), chosen_action_feature, chosen_reward_feature]) #128 + 5 + 5 

            logits = self.lin_actor(new_memory) ## 4
            if self.rl_algo_arg == 'A2C':
                value = self.lin_critic(new_memory)
                value = torch.squeeze(value)
            else:
                value = torch.empty(0)
            #action_probs = self.softmax_lyr(logits) ## 4

        else: 
            logits = torch.randn(self.action_dim)
            value = torch.empty(0)
            Wattn_topk_val = torch.empty(1,1,5)
            chosen_memory_ids = torch.empty(0)
            Wattn = torch.empty(1,1,5)

        return logits, value, torch.squeeze(Wattn_topk_val).detach().clone(), chosen_memory_ids, self.gate_alpha.detach().clone(), torch.squeeze(Wattn).detach().clone()
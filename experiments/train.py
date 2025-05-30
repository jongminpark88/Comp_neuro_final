# import os, sys
# # notebooks/에서 한 단계 위로 올라간 폴더를 PATH에 추가
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


# import os
# import yaml
# import torch
# import random
# import time
# from torch import nn, optim
# from torch.distributions import Categorical
# from env.custom_maze_env import CustomMazeEnv
# from env.get_retina_image import reconstruct
# from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
# import matplotlib.pyplot as plt
# import numpy as np
# from IPython import display


# #==========================
# # 모델 import 및 Load
# #==========================

# RNNPolicy = None


# #==========================
# # 메모리 import 및 Load
# #==========================

# RNNPolicy = None


# #==========================
# # set_seed
# #==========================
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


# def main():
#     # 1) 설정 불러오기
#     cfg = yaml.safe_load(open("experiments/configs/default.yaml"))
#     set_seed(cfg["train"]["seed"])

#     # 2) 환경 생성 & 래핑
#     base_env = CustomMazeEnv(**cfg["env"])
#     env = RGBImgPartialObsWrapper(base_env, tile_size=cfg["env"]["tile_size"])
#     obs_dim = np.prod(reconstruct(obs["image"], render_chanel=1).shape)  # flatten -> 6ox80
#     action_dim = env.action_space.n

#     # 3) 에이전트, 옵티마이저
#     policy = RNNPolicy(obs_dim, cfg["agent"]["hidden_size"], action_dim)
#     optimizer = optim.Adam(policy.parameters(), lr=cfg["agent"]["learning_rate"])
#     policy.train()

#     # tensorboard 준비
#     if cfg["logging"]["use_tensorboard"]:
#         from torch.utils.tensorboard import SummaryWriter
#         tb = SummaryWriter(cfg["logging"]["tensorboard_dir"])
#     else:
#         tb = None

#     # 4) 학습 루프
#     for ep in range(1, cfg["train"]["total_episodes"] + 1):
#         obs, _ = env.reset(seed=cfg["train"]["seed"] + ep)
#         # obs["image"] shape = (tile_size * view_size, tile_size * view_size, 3)
#         retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
#         # full_map = obs["image"] # 160 x 160 x 3


#         state = torch.from_numpy(retina).float().view(1,1,-1)
#         hx = torch.zeros(1, 1, cfg["agent"]["hidden_size"])
#         log_probs = []
#         rewards = []


#         #==========================
#         # 에피소드 학습
#         #==========================
#         done = False
#         while not done:
#             logits, hx = policy(state, hx)
#             m = Categorical(logits=logits)
#             a = m.sample()
#             log_probs.append(m.log_prob(a))

#             obs, r, term, trunc, _ = env.step(a.item())
#             rewards.append(r)

#             #======================================
#             ##### 메모리에 저장하는 코드 추가해야함 ###
#             # retina : retina(egocentric view)
#             # action : a
#             # hidden_state
#             # timestep : obs['timestep']
#             # agent_postion : obs['position']
#             # episode_number : ep
#             #========================================

#             # next state
#             retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
#             state = torch.from_numpy(retina).float().view(1,1,-1)
#             done = term or trunc



#         #============================
#         # 에피소드 당 손실 계산 (REINFORCE)
#         #============================
#         returns = []
#         G = 0
#         for r in reversed(rewards):
#             G = r + cfg["agent"]["gamma"] * G
#             returns.insert(0, G)
#         returns = torch.tensor(returns)
#         returns = (returns - returns.mean()) / (returns.std() + 1e-8)

#         loss = 0
#         for lp, G in zip(log_probs, returns):
#             loss -= lp * G

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


#         #============================
#         # 로그
#         #============================
#         ep_reward = sum(rewards)
#         if tb:
#             tb.add_scalar("train/episode_reward", ep_reward, ep)
#             tb.add_scalar("train/loss", loss.item(), ep)

#         if ep % cfg["train"]["log_interval"] == 0:
#             print(f"[Episode {ep}] reward={ep_reward:.2f}, loss={loss.item():.4f}")

#         if ep % cfg["train"]["save_interval"] == 0:
#             os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
#             torch.save(policy.state_dict(),
#                        os.path.join(cfg["train"]["checkpoint_dir"], f"policy_ep{ep}.pt"))

#     if tb:
#         tb.close()

# if __name__ == "__main__":
#     main()

import os, sys
# notebooks/에서 한 단계 위로 올라간 폴더를 PATH에 추가
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import yaml
import torch
import random
import time
from torch import nn, optim
from torch.distributions import Categorical
from env.custom_maze_env import CustomMazeEnv
from env.get_retina_image import reconstruct
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from model.policy import *

torch.autograd.set_detect_anomaly(True)

#==========================
# set_seed
#==========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(RL_ALGO_ARG):

    # 1) 설정 불러오기
    cfg = yaml.safe_load(open(os.path.join(project_root, "experiments/config/default.yaml"))) ##TBD - ERROR
    set_seed(cfg["train"]["seed"])

    # 2) 환경 생성 & 래핑
    #base_env = CustomMazeEnv(**cfg["env"]) ##TBD - ERROR
    base_env = CustomMazeEnv(**{'layout_id': 'e',
    'goal_pos': [0, 3],
    'view_size': 5,
    'max_steps': 1000,
    'tile_size': 32,
    'render_mode': 'rgb_array'})
    env = RGBImgPartialObsWrapper(base_env, tile_size=cfg["env"]["tile_size"])
    # obs_dim = np.prod(reconstruct(obs["image"], render_chanel=1).shape)  # flatten -> 6ox80
    action_dim = env.action_space.n

    # 3) 에이전트, 옵티마이저
    # hyperparameter
    eps_start = cfg["agent"]["eps_start"]
    eps_decay = cfg["agent"]["eps_decay"]
    eps = eps_start

    HIDDEN_SIZE = HIDDEN_SIZE = cfg["agent"]["hidden_size"]
    PARAMS = {
        'memory_bank_ep': {
            'decay_rate': 0.0001, 
            'noise_std': 0.001, 
            'et_lambda': 0.99,
            'memory_len': 5000,
            'update_freq': 100,
            'hidden_dim': HIDDEN_SIZE,
            'decay_yn': False
        },
        'cnn_embed': {
            'cnn_hidden_lyrs': [4, 8],
            'lin_hidden_lyrs': [512, HIDDEN_SIZE],
            'input_img_shape': (60, 80)
        },
        'rnn': {
            'input_size': HIDDEN_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'batch_first': True
        },
        'memory_gate': {
            'hidden_dim_lyrs': [HIDDEN_SIZE, int(HIDDEN_SIZE/2)],
            'action_dim': action_dim,
            'attn_size': 5,
            'rl_algo_arg': RL_ALGO_ARG
        }
    }

    # policy = RNNPolicy(obs_dim, cfg["agent"]["hidden_size"], action_dim)
    policy = cnnrnnattn_policy(PARAMS)
    optimizer = optim.Adam(policy.parameters(), lr=float(cfg["agent"]["learning_rate"]))
    memory_bank_ep = memory_bank(**PARAMS['memory_bank_ep'])
    hx = torch.randn(1, 1, HIDDEN_SIZE) / math.sqrt(HIDDEN_SIZE)

    # tensorboard 준비
    if cfg["logging"]["use_tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(cfg["logging"]["tensorboard_dir"], f'{RL_ALGO_ARG}')
        os.makedirs(log_dir, exist_ok=True)
        tb = SummaryWriter(log_dir)
    else:
        tb = None

    # 4) 학습 루프
    policy.train()
    for ep in range(1, cfg['train']['total_episodes']):
        
        obs, _ = env.reset(seed=cfg["train"]["seed"] + ep)
        # obs["image"] shape = (tile_size * view_size, tile_size * view_size, 3)
        retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
        # full_map = obs["image"] # 160 x 160 x 3
        state = torch.from_numpy(retina)[None, None, ...] ## 1 x 1 x 60 x 80

        log_probs, rewards, values = [], [], []
        gate_alpha_lst, terminated_lst = [], []

        #==========================
        # 에피소드 학습
        #==========================
        done = False
        while not done:
            logits, value, sx, hx, chosen_ids, gate_alpha_, attention_ = policy(state, hx, memory_bank_ep)
            m = Categorical(logits=logits)
            if random.random() < eps:
                a = torch.randint(action_dim, ())
            else:
                a = m.sample()
            log_probs.append(m.log_prob(a))
            values.append(value)
            obs, r, term, trunc, _ = env.step(a.item())
            rewards.append(r)

            memory_bank_ep.update(retina, sx.detach().clone(), hx.detach().clone(), a, r, obs, ep, chosen_ids)
            memory_bank_ep.save(cfg["logging"]["timestep_dir"], cfg["logging"]["attention_dir"], ep, obs['timestep'], chosen_ids, RL_ALGO_ARG)
            
            # next state
            retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
            state = torch.from_numpy(retina)[None, None, ...]
            done = term or trunc

            with torch.no_grad():
                hx = hx.detach().clone()
                gate_alpha_lst.append(gate_alpha_.item())
                terminated_lst.append(term)
                timestep_ = obs['timestep']
                os.makedirs(os.path.join(cfg["logging"]["attention_weight_dir"], f'ep{ep}'), exist_ok=True)
                torch.save(attention_,
                            os.path.join(cfg["logging"]["attention_weight_dir"], f'ep{ep}', f'attention_weight_{RL_ALGO_ARG}_{timestep_}.pt'))

            ### for checking
            #print(chosen_ids, gate_alpha_, r)

        #============================
        # 에피소드 당 손실 계산 (REINFORCE)
        #============================
        attn_size = PARAMS['memory_gate']['attn_size']
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + cfg["agent"]["gamma"] * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = returns[attn_size:]
        log_probs = torch.stack(log_probs[attn_size:])

        if RL_ALGO_ARG == 'REINFORCE':
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # loss = 0
            # for lp, G in zip(log_probs, returns):
            #     loss -= lp * G        
            
            loss = -(log_probs * returns).sum()

        elif RL_ALGO_ARG == 'A2C':
            values  = torch.stack(values[attn_size:])
            advantages = returns.detach() - values
            actor_loss = -(log_probs * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            alpha = cfg["train"]["actor_loss_coef"]
            loss = alpha * actor_loss + (1-alpha) * value_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        eps *= eps_decay

        #============================
        # 로그
        #============================
        ep_reward = sum(rewards)
        print(ep_reward)

        if tb:
            tb.add_scalar("train/episode_reward", ep_reward, ep)
            tb.add_scalar("train/loss", loss.item(), ep)

        if ep % cfg["train"]["log_interval"] == 0:
            print(f"[Episode {ep}] reward={ep_reward:.2f}, loss={loss.item():.4f}m algo={RL_ALGO_ARG}")

        if ep % cfg["train"]["save_interval"] == 0:
            time_now = dt.now().strftime("%Y-%m-%d-%H:%M")
            os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
            torch.save(policy.state_dict(),
                        os.path.join(cfg["train"]["checkpoint_dir"], f"policy_{RL_ALGO_ARG}_ep{ep}_{time_now}.pt"))
        
        os.makedirs(cfg["logging"]["gate_alpha_dir"], exist_ok=True)
        os.makedirs(cfg["logging"]["terminated_dir"], exist_ok=True)
        with open(os.path.join(cfg["logging"]["gate_alpha_dir"], f"gate_alpha_{RL_ALGO_ARG}_ep{ep}.pkl"), "wb") as file_gate_alpha:
            pickle.dump(gate_alpha_lst, file_gate_alpha)
        with open(os.path.join(cfg["logging"]["terminated_dir"], f"terminated_{RL_ALGO_ARG}_ep{ep}.pkl"), "wb") as file_terminated:
            pickle.dump(terminated_lst, file_terminated)

        if tb:
            tb.close() 

if __name__ == "__main__":
    RL_ALGO_ARG = sys.argv[1]
    main(RL_ALGO_ARG)
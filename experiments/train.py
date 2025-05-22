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


#==========================
# 모델 import 및 Load
#==========================

RNNPolicy = None


#==========================
# 메모리 import 및 Load
#==========================

RNNPolicy = None


#==========================
# set_seed
#==========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # 1) 설정 불러오기
    cfg = yaml.safe_load(open("experiments/configs/default.yaml"))
    set_seed(cfg["train"]["seed"])

    # 2) 환경 생성 & 래핑
    base_env = CustomMazeEnv(**cfg["env"])
    env = RGBImgPartialObsWrapper(base_env, tile_size=cfg["env"]["tile_size"])
    obs_dim = np.prod(env.observation_space["image"].shape)  # flatten -> 6ox80
    action_dim = env.action_space.n

    # 3) 에이전트, 옵티마이저
    policy = RNNPolicy(obs_dim, cfg["agent"]["hidden_size"], action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["agent"]["learning_rate"])


    # tensorboard 준비
    if cfg["logging"]["use_tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(cfg["logging"]["tensorboard_dir"])
    else:
        tb = None

    # 4) 학습 루프
    for ep in range(1, cfg["train"]["total_episodes"] + 1):
        obs, _ = env.reset(seed=cfg["train"]["seed"] + ep)
        # obs["image"] shape = (tile_size * view_size, tile_size * view_size, 3)
        retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
        # full_map = obs["image"] # 160 x 160 x 3


        state = torch.from_numpy(retina).float().view(1,1,-1)
        hx = torch.zeros(1, 1, cfg["agent"]["hidden_size"])
        log_probs = []
        rewards = []


        #==========================
        # 에피소드 학습
        #==========================
        done = False
        while not done:
            logits, hx = policy(state, hx)
            m = Categorical(logits=logits)
            a = m.sample()
            log_probs.append(m.log_prob(a))

            obs, r, term, trunc, _ = env.step(a.item())
            rewards.append(r)

            #======================================
            ##### 메모리에 저장하는 코드 추가 ###
            # retina : retina(egocentric view)
            # action : a
            # hidden_state
            # timestep : obs['timestep']
            # agent_postion : obs['position']
            # episode_number : ep
            #========================================

            # next state
            retina = reconstruct(obs["image"], render_chanel=1)# 60 x 80
            state = torch.from_numpy(retina).float().view(1,1,-1)
            done = term or trunc



        #============================
        # 에피소드 당 손실 계산 (REINFORCE)
        #============================
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + cfg["agent"]["gamma"] * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, G in zip(log_probs, returns):
            loss -= lp * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #============================
        # 로그
        #============================
        ep_reward = sum(rewards)
        if tb:
            tb.add_scalar("train/episode_reward", ep_reward, ep)
            tb.add_scalar("train/loss", loss.item(), ep)

        if ep % cfg["train"]["log_interval"] == 0:
            print(f"[Episode {ep}] reward={ep_reward:.2f}, loss={loss.item():.4f}")

        if ep % cfg["train"]["save_interval"] == 0:
            os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
            torch.save(policy.state_dict(),
                       os.path.join(cfg["train"]["checkpoint_dir"], f"policy_ep{ep}.pt"))

    if tb:
        tb.close()

if __name__ == "__main__":
    main()


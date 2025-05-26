# envs/custom_maze_map.py
import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal


# ---------- 실제 미로 레이아웃 (0=통로, 1=벽) ----------
LAYOUTS = {
    "a": np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1],
        [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1],
        [1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
        [1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1],
        [1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1],
        [1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ]),

    "b": np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1],
        [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
        [1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
        [1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,1],
        [1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ]),

    "c": np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
        [1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1],
        [1,1,1,0,1,1,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1],
        [1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1],
        [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,0,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1],
        [1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
        [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ]),
}

class CustomMazeEnv(MiniGridEnv):
    """
    - 고정 미로 + 랜덤 시작(통로 위 전부) + 고정 목표
    - obs: dict(view, timestep, position)
    - reward: goal 도달 시 1, 그 외 0
    """

    def __init__(
        self,
        layout_id: str = "c",
        goal_pos=(0, 3),
        view_size: int = 5,
        max_steps: int = 1000,
        tile_size: int = 32,
        render_mode: str = "rgb_array",
        train_mode = 'train'
    ):
        # 1) 레이아웃 & 목표
        self.layout    = LAYOUTS[layout_id]
        self.goal_pos  = tuple(goal_pos)
        self.layout_id = layout_id
        grid_size      = self.layout.shape[0]

        # 2) 이 레이아웃의 모든 '통로' 셀을 안전 시작 후보로 저장
        #    (goal_pos 는 제외)
        ys, xs = np.where(self.layout == 0)
        self.safe_starts = [
            (int(x), int(y))
            for y, x in zip(ys, xs)
            if (y, x) != self.goal_pos
        ]

        # train_mode에 따라 90% vs 10% 분할
        n = len(self.safe_starts)
        cutoff = int(0.9 * n)

        if train_mode == 'train':
            # 출발 지점 전체의 90%
            self.safe_starts = self.safe_starts

        else:
            # 출발 지점 전체의 10%
            self.safe_starts = self.safe_starts


        # 3) MissionSpace
        mission_space = MissionSpace(
            mission_func=lambda **kwargs: "reach the green goal square"
        )

        # 4) 부모 초기화
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            agent_view_size=view_size,
            tile_size=tile_size,
            render_mode=render_mode,
        )

    def _gen_grid(self, width, height):
        # 1) 그리드 초기화
        self.grid = Grid(width, height)

        # 2) 벽 배치
        for y in range(height):
            for x in range(width):
                if self.layout[y, x] == 1:
                    self.grid.set(x, y, Wall())

        # 3) 목표 배치
        self.grid.set(*self.goal_pos, Goal())

        # 4) 안전한 시작 좌표에서 무작위 선택
        idx = self._rand_int(0, len(self.safe_starts))
        self.agent_pos = self.safe_starts[idx]

        # 5) 방향 랜덤
        self.agent_dir = self._rand_int(0, 4)

        # 6) 미션 문자열
        self.mission = "reach the green goal square"

    def reset(self, *, seed=None, options=None):
        """
        - 부모 reset 호출하여 _gen_grid → agent 배치까지 진행
        - 이후 obs dict 로 포장
        """
        obs_raw, info = super().reset(seed=seed, options=options)
        obs = {
            "view":     obs_raw,
            "timestep": 0,
            "position": tuple(self.agent_pos),
        }
        return obs, info

    def step(self, action):
        obs_raw, reward, terminated, truncated, info = super().step(action)

        reward = reward - 0.01

        # goal 도달 시 보상/종료
        if tuple(self.agent_pos) == self.goal_pos:
            reward = reward + 100
            terminated = True
            
        obs = {
            "view":     obs_raw,
            "timestep": self.step_count,
            "position": tuple(self.agent_pos),
        }
        return obs, reward, terminated, truncated, info

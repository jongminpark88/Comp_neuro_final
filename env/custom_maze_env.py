# envs/custom_maze_map.py
import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal


# ---------- 실제 미로 레이아웃 (0=통로, 1=벽) ----------
LAYOUTS = {
    "a": np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,1,0,0,0,1,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,0,1,1,1],
        [1,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1],
    ]),
    "b": np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,1,0,0,0,1,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,0,1,1,1],
        [1,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1],
    ]),
    "c": np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,1,0,0,0,1,0,1],
        [1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1],
    ]),
}


class CustomMazeEnv(MiniGridEnv):
    """
    - 고정 미로 + 랜덤 시작 + 고정 목표
    - obs: dict(view, timestep, position)
    - reward: goal 도달 시 1, 그 외 0
    """

    def __init__(
        self,
        layout_id: str = "a",
        goal_pos=(7, 1),
        view_size: int = 5,
        max_steps: int = 250,
        tile_size: int = 32,
        render_mode: str = "rgb_array",   # ← render_mode 기본값 설정
    ):
        # 레이아웃 & 목표
        self.layout   = LAYOUTS[layout_id]
        self.goal_pos = tuple(goal_pos)
        grid_size     = self.layout.shape[0]

        # MissionSpace
        mission_space = MissionSpace(
            mission_func=lambda **kwargs: "reach the green goal square"
        )

        # 부모 초기화 (render_mode 포함)
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            agent_view_size=view_size,
            tile_size=tile_size,
            render_mode=render_mode,       # ← 여기에 넘겨줌
        )

    def reset(self, *, seed=None, options=None):
        """
        reset 시에도 step()과 같은 obs dict 반환
        """
        obs_raw, info = super().reset(seed=seed, options=options)
        obs = {
            "view":     obs_raw,
            "timestep": 0,
            "position": tuple(self.agent_pos),
        }
        return obs, info

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # 벽 배치
        for y in range(height):
            for x in range(width):
                if self.layout[y, x] == 1:
                    self.grid.set(x, y, Wall())

        # 목표 배치
        self.grid.set(*self.goal_pos, Goal())

        # 시작 위치 (통로 중 랜덤)
        free_cells = [
            pos
            for pos in zip(*np.where(self.layout == 0))
            if pos != self.goal_pos
        ]

        idx           = self._rand_int(0, len(free_cells))
        self.agent_pos = tuple(free_cells[idx])

        # 방향 랜덤
        self.agent_dir = self._rand_int(0, 4)

        # 미션 문자열
        self.mission = "reach the green goal square"

    def step(self, action):
        obs_raw, reward, terminated, truncated, info = super().step(action)

        # goal 도달 보상/종료
        if tuple(self.agent_pos) == self.goal_pos:
            reward     = 1.0
            terminated = True
        else:
            reward     = 0.0

        obs = {
            "view":     obs_raw,
            "timestep": self.step_count,
            "position": tuple(self.agent_pos),
        }
        return obs, reward, terminated, truncated, info

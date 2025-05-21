## input : 1인칭 top-down view
## output : 1인칭 ego-centric view(60x80 3채널-RGB or 60x80 1채널-GRAY)

import numpy as np
import matplotlib.pyplot as plt

# 색 팔레트: [sky, wall, floor]
COLORS = np.array([
    [135, 206, 235],  # sky
    [150, 150, 150],  # wall
    [ 60,  60,  60],  # floor
], dtype=np.float32) / 255.0

def reconstruct(fp_view,
                n_rays=80,    # 가로 80픽셀 (2° 간격)
                fov_h=160,    # 수평 시야각 160°
                fov_v=120,    # 수직 시야각 120°
                eye_h=180,    # 눈 높이(cm)
                wall_h=250,   # 벽 높이(cm)
                px_per_cm=1/5,
                d_max_cm=300,
                wall_value=146,
                render_mode=False,
                render_chanel=3):
    
    """
    fp_view: (160,160,3) RGB 1인칭 뷰
    render_chanel: 3 → RGB, 1 → Gray
    """
    # 0) 출력 해상도
    out_h, out_w = 60, 80

    H, W, _ = fp_view.shape
    sy, sx  = 143, 79   # 에이전트 발끝 픽셀 좌표

    # 그레이스케일로 변환
    gray = fp_view.mean(2).astype(int)

    # 1) Ray 방향 벡터 준비
    thetas = np.deg2rad(np.linspace(-fov_h/2, fov_h/2, n_rays))
    dirs   = np.stack([np.sin(thetas), -np.abs(np.cos(thetas))], 1)

    # 히트 포인트 저장용
    hit_xy = np.zeros((n_rays, 2), int)
    d_px   = np.zeros(n_rays)

    # 2) RayCast + “연속 10픽셀 벽” 검사
    for i, (vx, vy) in enumerate(dirs):
        ts = [
            (0 - sy) / vy if vy < 0 else np.inf,
            (0 - sx) / vx if vx < 0 else np.inf,
            ((W-1) - sx) / vx if vx > 0 else np.inf
        ]
        t_hit = min(t for t in ts if t > 0)

        consec = 0
        step   = 0.0
        found  = False
        start  = 0.0

        while step <= t_hit:
            y = int(round(sy + vy * step))
            x = int(round(sx + vx * step))

            if gray[y, x] == wall_value:
                if consec == 0:
                    start = step
                consec += 1
                if consec >= 10:
                    # 연속 10픽셀을 모두 wall_value로 만났다면 hit
                    y0, x0 = int(round(sy + vy * start)), int(round(sx + vx * start))
                    hit_xy[i] = (x0, y0)
                    d_px[i]   = start
                    found = True
                    break
            else:
                consec = 0

            step += 1.0

        if not found:
            # 벽을 만나지 못하면 경계 지점에서 강제종료
            y0, x0 = int(round(sy + vy * t_hit)), int(round(sx + vx * t_hit))
            hit_xy[i] = (x0, y0)
            d_px[i]   = t_hit

    # 3) 픽셀→cm 변환 & 각도 계산
    d_cm = (d_px / px_per_cm) * np.abs(np.cos(thetas))
    d_cm = np.clip(d_cm, 0, d_max_cm)
    d_m  = d_cm / 100.0  # 미터 단위

    theta_floor = np.arctan2(-eye_h/100.0,        d_m)
    theta_wall  = np.arctan2((wall_h-eye_h)/100.0, d_m)

    # 4) 각도→화면 row 인덱스
    fov_v_rad = np.deg2rad(fov_v)
    top, bot  = fov_v_rad/2, -fov_v_rad/2
    ang2row   = lambda th: np.clip(((top - th)/fov_v_rad * out_h).astype(int), 0, out_h-1)

    sky_r   = ang2row(theta_wall)
    floor_r = ang2row(theta_floor)

    # 5) 이상치 제거 (좌우 3-픽셀 median smoothing)
    sky_s   = sky_r.copy()
    floor_s = floor_r.copy()
    for j in range(1, out_w-1):
        sky_s[j]   = np.median(sky_r[j-1:j+2])
        floor_s[j] = np.median(floor_r[j-1:j+2])
    sky_r, floor_r = sky_s, floor_s

    # 6) 60×80 재구성 (히트 y 기준 음영)
    img = np.zeros((out_h, out_w, 3), np.float32)
    for c in range(n_rays):
        s, f = sky_r[c], floor_r[c]
        hy   = hit_xy[c, 1]
        shade = np.clip(hy / sy, 0.0, 1.0)

        img[0:s  , c] = COLORS[0]           # sky
        img[s:f  , c] = COLORS[1] * shade   # wall
        img[f:   , c] = COLORS[2]           # floor

    # 7) 디버그: 레이 + 히트 지점
    if render_mode:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(fp_view)
        ax.scatter([sx],[sy], c='cyan', s=60, label='agent')
        for (hx, hy) in hit_xy:
            ax.plot([sx, hx], [sy, hy], c='yellow', lw=1)
        ax.scatter(hit_xy[:,0], hit_xy[:,1], c='red', s=20, label='hit')
        ax.axis('off'); ax.legend(); plt.show()

    # ★★★★★ 여기부터 ★★★★★
    if render_chanel == 1:
        # grayscale 변환 (NTSC 가중치)
        gray_img = (
            img[...,0] * 0.2989 +
            img[...,1] * 0.5870 +
            img[...,2] * 0.1140
        )
        return gray_img  # shape = (60,80)

    # 기본은 RGB
    return img  # shape = (60,80,3)



'''
# 사용 예
# fp_view = obs["image"] 로 얻은 (160,160,3) 어레이
recon = reconstruct(fp_view)
plt.figure(figsize=(6,4))
plt.imshow(recon)
plt.axis("off")
plt.show()
'''
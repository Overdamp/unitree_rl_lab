# G1 23-DoF: เชื่อม unitree_rl_lab (Isaac) กับ unitree_rl_mjlab (MuJoCo)

เริ่มแผน 28/10/2026 · เจ้าของงาน: Luke

## เป้าหมาย

เทรน policy G1 23-DoF ได้ทั้งใน Isaac Lab (PhysX) และ mjlab (MuJoCo-Warp) โดยใช้ "สัญญากลาง" ชุดเดียวคือ `policy.onnx` + `deploy.yaml` แล้วรันด้วย `g1_ctrl` ตัวเดียวทั้งใน `unitree_mujoco` และหุ่นจริง พร้อมใช้การทดสอบข้าม engine (sim2sim) เป็นด่านก่อนขึ้นหุ่นจริง

```
unitree_rl_lab (Isaac)  ──export──▶  policy.onnx + deploy.yaml  ──▶  g1_ctrl ──▶ unitree_mujoco / หุ่นจริง
unitree_rl_mjlab (MuJoCo) ─export──▶  policy.onnx + deploy.yaml  ──▶  g1_ctrl
```

## Source of truth ของค่าหุ่น

ค่าหุ่นยึดตาม `unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_23dof_constants.py` (kp = armature·ω², kd = 2ζ·armature·ω, ω = 10 Hz, ζ = 2)

| กลุ่ม | joint | kp | kd | armature | effort | action scale |
|---|---|---|---|---|---|---|
| 5020 | ไหล่, ศอก, wrist roll | 14.25 | 0.907 | 0.00361 | 25 | 0.439 |
| 7520-14 | hip pitch/yaw, waist yaw | 40.18 | 2.558 | 0.01018 | 88 | 0.548 |
| 7520-22 | hip roll, knee | 99.10 | 6.309 | 0.02510 | 139 | 0.351 |
| ankle (2×5020) | ankle pitch/roll | 28.50 | 1.814 | 0.00722 | 50 | 0.439 |

ท่ายืน: hip pitch −0.1, knee 0.3, ankle pitch −0.2, shoulder pitch 0.35, shoulder roll ±0.18, elbow 0.87

## Implement list

| # | งาน | repo | กำหนด | สถานะ |
|---|---|---|---|---|
| 0 | ติดตั้ง Isaac Sim กลับเข้า `~/nvidia/IsaacLab/.venv` (`uv sync --extra isaacsim`) | IsaacLab | 28/10/2026 | ✅ Isaac Sim 6.2.0.0 |
| 0.1 | ⚠️ พบว่า unitree_rl_lab (branch `isaaclab3.0`, commit `2522f89` 25/05/2026) **ใช้กับ IsaacLab v3.0.0 release ไม่ได้**: ยัง import `isaaclab_tasks.manager_based...` และใช้ reward `track_lin_vel_xy_yaw_frame_exp` ที่ถูกย้าย/ลบใน commit `ea8ecd6ef` (01/06/2026) ต้องเลือก: (A) ใช้ IsaacLab commit `207f319d9` (ก่อนแยก core/contrib, ใช้ Isaac Sim 6.0.0 ตรงกับ README ของ unitree_rl_lab) ใน worktree แยก หรือ (B) port unitree_rl_lab ไป IsaacLab 3.0 (ต้องแก้ไฟล์ที่ใช้ร่วมทุกหุ่น) | IsaacLab / unitree_rl_lab | 28/10/2026 | ✅ ติดตั้งแล้ว (Isaac Sim 6.0.0.0, torch 2.10.0+cu128, rsl-rl-lib 5.0.1, isaaclab 6.1.0) เลือก (A): worktree `~/nvidia/IsaacLab-unitree` @ `207f319d9` + venv `env_isaaclab` ในโฟลเดอร์นั้น (Isaac Sim 6.0.0, torch 2.10 cu128, rsl-rl) |
| 1 | เพิ่ม task `Unitree-G1-23dof-Velocity` และ `-Flat` (ใช้ MDP ของ 29dof ผ่าน subclass) | unitree_rl_lab | 28/10/2026 | ✅ เขียนโค้ดแล้ว |
| 2 | ตั้งค่า actuator, armature, ท่ายืน, action scale ให้ตรงกับ mjlab | unitree_rl_lab | 28/10/2026 | ✅ เขียนโค้ดแล้ว |
| 2.1 | Smoke test: `list_envs`, เทรน 16 env × 5 iter, เช็ค `params/deploy.yaml` ว่า kp/kd/scale ตรงตารางข้างบน | unitree_rl_lab | 28/10/2026 | ⏳ |
| 3 | เทรน Flat บนเครื่องนี้ (≤1024 env เพราะ VRAM 6 GB) ดู reward ว่าขึ้น; เช็คว่า DGX (aarch64) รัน Isaac Sim ได้ไหม | unitree_rl_lab | 29–30/10/2026 | ⏳ |
| 3.1 | เทรน Rough เต็ม (DGX ถ้ารองรับ ไม่งั้นเครื่องนี้ค้างคืน) | unitree_rl_lab | 31/10–02/11/2026 | ⏳ |
| 4 | ตรวจความเข้ากันของ `deploy.yaml` ระหว่างสองสาย: ความยาว/ลำดับของ `stiffness`, `damping`, `default_joint_pos`, `joint_ids_map` (unitree_rl_lab export เป็น SDK order 27 ช่อง ส่วน mjlab เป็น policy order 23 ช่อง) แล้วแก้ที่ฝั่ง export หรือเขียนตัวแปลง | ทั้งสอง | 03/11/2026 | ⏳ |
| 4.1 | วาง ONNX + yaml ที่ `unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/velocity/isaac_v0/` แล้วรันด้วย `g1_ctrl` + `unitree_mujoco` วัดความเร็ว vx 0.4/0.8/1.0 เทียบกับ P3b | unitree_rl_mjlab | 04/11/2026 | ⏳ |
| 5 | ทางกลับ: เล่น P3b (mjlab) ใน Isaac เพื่อเช็คความทนข้าม engine | unitree_rl_lab | 05/11/2026 | ✅ `scripts/play_deploy_onnx.py` (วิธี A ด้านล่าง) · 02/11: P3b, yaw_stand บน PhysX / Newton-MJWarp / MuJoCo ต่างกัน ~±8 pp ไม่ล้ม · 16/11 Newton: vx 0.8 → 0.58 m/s (MuJoCo vel_eval 0.59) |
| 6 | สรุปผลลง doc + เลือก policy ที่ผ่านทั้งสอง engine ไปทดสอบบน gantry | ทั้งสอง | 06/11/2026 | ⏳ |

## ข้อควรระวัง

- `~/nvidia/IsaacLab` (v3.0.0) ใช้กับ unitree_rl_lab ไม่ได้ ใช้ `~/nvidia/IsaacLab-unitree` เท่านั้น
- รันด้วย `source ~/nvidia/IsaacLab-unitree/env_isaaclab/bin/activate` แล้ว `python scripts/rsl_rl/train.py` ไม่ใช่ `./unitree_rl_lab.sh` (สคริปต์นั้นใช้ python จาก `$CONDA_PREFIX`)
- MDP ของ unitree_rl_lab ต่างจาก mjlab (obs scale 0.2/0.05, history 5, reward) จึงเทียบ "ค่าหุ่น" ได้ตรงกัน แต่ policy ไม่เหมือนกัน; `deploy.yaml` เป็นตัวบอก `g1_ctrl` ว่าต้องประกอบ obs อย่างไร
- `deploy/robots/g1_23dof/config/config.yaml` ของ unitree_rl_lab ยังใช้ kp FixStand ชุดเดิม (100/150/200/40) ห้ามสลับไปใช้กับหุ่นจริงก่อนตรวจขั้น 4
- ห้ามแก้ไฟล์ที่ใช้ร่วมหลายหุ่น (`unitree.py`, `export_deploy_cfg.py`, task 29dof) ให้แก้เฉพาะโฟลเดอร์ `g1/23dof`

## แผนระยะยาว: sensor ใน Isaac และเทรนต่อข้าม engine (16–17/11/2026)

### สองวิธีเล่น policy ของ mjlab ใน Isaac

| | A: deploy ONNX (`scripts/play_deploy_onnx.py`) | B: task ใน Isaac ที่ MDP ตรงกับ mjlab |
|---|---|---|
| หลักการ | Isaac เป็นแค่ physics; สคริปต์ประกอบ obs ตาม `deploy.yaml` และเรียงข้อต่อเองแบบ `g1_ctrl` | ตั้ง observation/action manager ของ Isaac ให้เหมือน mjlab แล้วใช้ `scripts/rsl_rl/play.py` / `train.py` ตามปกติ |
| ไฟล์ | `policy.onnx` + `deploy.yaml` | `model.pt` |
| ทดสอบอะไร | สัญญา deploy (จับ `deploy.yaml` ผิดได้) | policy ใน env ของ Isaac (หลาย env, terrain, event) |
| เทรน / fine-tune ใน Isaac | ไม่ได้ | ได้ |
| รองรับตอนนี้ | blind/flat MLP, หุ่นตัวเดียว, พื้นเรียบ | ยังไม่มี |

**ทำไม `play.py` โหลด P3b ตรงๆ ไม่ได้:** task `Unitree-G1-23dof-Velocity(-Flat)` (subclass ของ 29dof) มี obs ต่อ frame 78 (ไม่มี `gait_phase`) × history 5 = 390 แต่ P3b รับ 80 (มี `gait_phase`, history 1) · scale ang_vel/joint_vel 0.2/0.05 · Isaac ปิด obs normalizer แต่ checkpoint มี `obs_normalizer.*` · ลำดับข้อต่อ MuJoCo (XML) ≠ Isaac → `size mismatch for mlp.0.weight: [512, 80] vs [512, 390]` · ที่ตรงกันแล้ว: rsl-rl-lib 5.0.1 ทั้งสองฝั่ง, actor 512-256-128

### sensor ใน Isaac Lab เทียบ mjlab

| mjlab (ใช้อยู่ในงานบันได) | Isaac Lab | หมายเหตุ |
|---|---|---|
| `RayCastSensorCfg` height scan 17×11 (teacher) | `RayCaster` + `GridPatternCfg` | เทียบกันได้ตรง |
| ray ลงจาก ankle roll link (foot clearance) | `RayCaster` ติด body | เทียบกันได้ตรง |
| กล้อง depth D455 จำลอง 24×40 (student) | `TiledCamera` depth | renderer RTX ต่างจาก MuJoCo → ทดสอบความทนของ student ได้ |
| ไม่มี | RGB สมจริง, RTX lidar | ข้อได้เปรียบของ Isaac |

ข้อจำกัด: RTX 3060 6 GB เทรนด้วยกล้องจำนวนมากไม่ไหว (ต้องใช้ DGX) · ต้องเช็กว่า Isaac Sim รองรับ DGX Spark (aarch64) พร้อม renderer กล้องไหม · Isaac Sim 6.0.0 ใน `env_isaaclab` Kit GUI/video ยังพัง กล้องต้องทดสอบก่อน

### เทรนต่อข้าม engine (resume `model.pt` ระหว่าง mjlab ↔ Isaac)

ทำได้ (rsl-rl เดียวกัน, network เดียวกัน) ถ้าทุกข้อตรงกัน:

| ต้องตรง | ถ้าไม่ตรง |
|---|---|
| obs ของ actor: term, ลำดับ, scale, history, normalizer | โหลดไม่ผ่าน หรือโหลดได้แต่ policy พัง |
| obs ของ critic | ต้องทิ้ง critic แล้วเริ่มใหม่ |
| ลำดับข้อต่อ, action scale, ท่า default | สั่งผิดข้อ |
| kp/kd, armature, effort, dt 0.005 × decimation 4 | การตอบสนองต่างไป |
| sensor: ตำแหน่ง, pattern ray, ความละเอียด/FOV กล้อง, delay | obs มีความหมายต่างไป |
| reward term/weight | ไม่บังคับ แต่ critic ประเมินผิดช่วงแรก |

หลังสลับ engine: ผลตกชั่วคราวจาก contact/physics ต่างกัน → warm-up critic หรือลด learning rate ช่วงแรก · ได้ domain randomization ข้าม engine

### แนวทางที่เลือก: สัญญา MDP กลาง + ทำทีละขั้น

```
              สัญญา MDP กลาง (1 ไฟล์): obs / action / ข้อต่อ / PD / sensor / history
        ┌────────────────────┼─────────────────────┐
   mjlab task           Isaac task              deploy.yaml → g1_ctrl
 (เทรนหลัก, DGX)    (ตรวจข้าม engine / sensor)      (หุ่นจริง)
```

| ขั้น | งาน | เงื่อนไขเริ่ม | สถานะ |
|---|---|---|---|
| L0 | เทรนใน mjlab ต่อ (stairs teacher4 → depth student) ตรวจข้าม engine ด้วยวิธี A | — | ⏳ teacher4 กำลังเทรน (16/11) |
| L1 | ขยาย `deploy.yaml` เป็นสัญญากลาง: เพิ่มสเปก sensor (ตำแหน่ง/FOV/ความละเอียดกล้อง, pattern ray), history และ delay ของแต่ละ term · ทุก obs ใหม่ใน mjlab ต้องลงสัญญานี้ | ทำได้เลย (งานเล็ก) | ⏳ |
| L2 | ขยายวิธี A สำหรับ depth student: กล้อง depth ใน Isaac ตามสัญญา, history 3 + delay, `OnnxMlp` รองรับ `Conv`, ฉากบันได open-riser ใน Isaac | student ผ่านเป้าใน mjlab | ⏳ |
| L3 | วิธี B: สร้าง task Isaac จากสัญญากลาง (โฟลเดอร์ `g1/23dof` เท่านั้น) → `play.py` โหลด `model.pt` ของ mjlab ได้ → fine-tune ข้าม engine / ใช้ sensor ที่ mjlab ไม่มี | L2 ผ่าน และมีเหตุผลต้องใช้ Isaac เป็นสนามฝึก | ⏳ |

ทำไมยังไม่ย้ายการเทรนไป Isaac: mjlab บน DGX เร็ว (~14k steps/s) และมี pipeline บันไดครบ (terrain open-riser, reward, curriculum, eval, deploy) · กล้อง depth ของ mjlab พอสำหรับ student · ย้ายตอนนี้ต้องสร้างทั้งหมดใหม่ใน Isaac โดยยังไม่รู้ว่าจำเป็น


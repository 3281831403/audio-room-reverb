# audio-room-reverb

## 题面

给关卡自动生成房间混响后，小房间像大教堂、混响还抢在直达声前面：RT60 只用体积，忽略表面积，小房间混响过长；早期反射不做延迟排序，混响先于直达声；镜像源不限制阶数，指数爆炸；尾音不按 RT60 截断，固定 5 秒；卷积输出长度算错，尾巴被截断 1 个 IR 长度；干湿比不钳制，ratio=2 时总电平翻倍；距离衰减在 ref 处增益为 0.5，整体偏小 6 dB；材料吸收系数不钳制到 0..1，负值产生增益；批量共用 IR，房间之间串；采样率不一致不校验，卷积后音调偏移。reverb/room.py 是核心，tests/test_room.py 15个红。要修的是：RT60 同时使用体积与表面积；早期反射按延迟排序；镜像源限制阶数；尾音按 RT60 截断；修正卷积输出长度；干湿比钳制到 0..1；距离衰减在 ref 处增益为 1；吸收系数钳制到 0..1；每个房间独立生成 IR；校验采样率一致并在任一轨非法时整体回滚。

## 运行方式

从仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始失败环境的输出尾部应为 `15 failed`（例如 `15 failed in 0.xxs`），且没有 collection error、import error 或 syntax error。15 条测试都会直接调用未实现的桩，因此每条均以 `NotImplementedError` 失败。实现只允许改动 `reverb/room.py`，完成后同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_room.py` 是固定评测用例，不可修改。`conftest.py` 只负责把仓库根加入 `sys.path`；`.gitignore` 仅忽略 Python/pytest 缓存。实现仅使用 Python 标准库（`math`、`cmath`），不联网、不使用随机数或 `sleep`，噪声必须由显式种子的确定性哈希生成。数值容差统一为 `1e-6`。

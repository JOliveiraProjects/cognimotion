# CognitiveMotionIntelligence — Python Layer

## Structure

```
motion_intelligence/
├── config.py                          # All config dataclasses
├── data/
│   ├── pose_frame.py                  # PoseFrame, Trajectory, TrajectorySample
│   └── trajectory_frame.py            # TrajectoryFrame (past+future pair)
├── encoding/
│   ├── pose_encoder.py                # Transformer-based pose encoder (256-dim)
│   ├── trajectory_encoder.py          # Trajectory encoder (128-dim)
│   └── motion_latent_space.py         # VAE latent space with style conditioning
├── memory/
│   ├── motion_memory_bank.py          # FAISS similarity search bank
│   └── motion_replay_buffer.py        # Prioritized replay (diversity+recency+conf)
├── learning/
│   ├── online_imitation_learner.py    # AdamW + CosineAnnealing online learner
│   └── dream_scheduler.py             # Async imagination/dreaming loop
├── protocol/
│   └── binary_protocol.py             # Binary protocol compatible with UE5 C++
├── runtime/
│   └── motion_inference_service.py    # Async TCP server (asyncio)
└── utils/
    ├── logger.py                       # Structured logger
    ├── metrics.py                      # Quality metrics (foot sliding, smoothness)
    └── pose_legality_validator.py      # Constraint-based pose validator
```

## Requirements

```
pip install -r requirements.txt
```

## Running the Server

```python
from motion_intelligence.runtime.motion_inference_service import main
main()
```

Or directly:

```
python -m motion_intelligence.runtime.motion_inference_service
```

## Configuration

```python
from motion_intelligence.config import MotionIntelligenceConfig, ServerConfig

config = MotionIntelligenceConfig(
    server=ServerConfig(host="0.0.0.0", port=9000, worker_threads=4),
    device="cuda",  # or "cpu"
)
```

## UE5 Connection

The server listens on `127.0.0.1:9000` by default.  
In UE5, assign `UCognitiveMotionLearnerComponent` to your NPC and set `PythonHost` + `PythonPort`.

## Protocol

Binary protocol with 24-byte header:  
`magic(4) + version(1) + msg_type(1) + flags(2) + payload_size(4) + checksum(4) + seq_id(8)`

Message types: `0x10` Handshake, `0x11` HandshakeAck, `0x01` MotionRequest, `0x02` MotionResponse, `0x03` PoseFrame.

## Quality Metrics

- **foot_sliding**: deviation of foot position from expected during movement (< 0.1 = good)
- **smoothness**: jerk-based metric from velocity sequence (> 0.8 = good)
- **imitation_score**: how closely the selected animation matches the target trajectory
- **trajectory_error**: L2 distance between predicted and actual trajectory
- **confidence**: embedding confidence from the pose encoder [0, 1]

"""
protocol/binary_protocol.py
============================
CORREÇÕES APLICADAS NESTA VERSÃO:
  CP-01: parse_leader_sequence — adicionados StartTimestamp + EndTimestamp (16 bytes)
         que o C++ SerializeLeaderSequence escreve e que estavam completamente ausentes.
         Sem isso Python lia 4 bytes de StartTimestamp como n_frames → crash garantido.
  CP-02: parse_leader_sequence — SequenceId mudado de "<i" (LE) para ">I" (BE)
         para espelhar WriteInt32 do C++ que usa BYTESWAP_ORDER32 (big-endian).
  CP-03: build_leader_sequence — mesmo fix bilateral: adiciona StartTimestamp/EndTimestamp
         e corrige SequenceId para ">I".

CORREÇÕES ANTERIORES MANTIDAS:
  BUG-3: parse_motion_request — raw_pose_payload passado diretamente para parse_pose_frame().
  BUG-7: parse_autonomous_request — header reconstruído com msg_type=MSG_MOTION_REQUEST.
"""
from __future__ import annotations
import struct
import numpy as np
from typing import Optional, Tuple
from data.pose_frame import PoseFrame, Trajectory, TrajectorySample

MAGIC_HEADER     = 0x434D4900
PROTOCOL_VERSION = 1
HEADER_FORMAT    = "<IBBHIIq"
HEADER_SIZE      = struct.calcsize(HEADER_FORMAT)

MSG_MOTION_REQUEST     = 0x01
MSG_MOTION_RESPONSE    = 0x02
MSG_POSE_FRAME         = 0x03
MSG_LEADER_SEQUENCE    = 0x05
MSG_AUTONOMOUS_REQUEST = 0x06
MSG_MOTION_ACTION      = 0x07
MSG_PERCEPTION         = 0x08
MSG_TEACH              = 0x09
MSG_HANDSHAKE          = 0x10
MSG_HANDSHAKE_ACK      = 0x11
MSG_PING               = 0x20
MSG_PONG               = 0x21
MSG_ERROR              = 0xFF


def _crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return crc ^ 0xFFFFFFFF


def _build_frame(msg_type: int, seq_id: int, payload: bytes) -> bytes:
    checksum = _crc32(payload) if payload else 0
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_HEADER, PROTOCOL_VERSION, msg_type,
        0, len(payload), checksum, seq_id,
    )
    return header + payload


def _parse_header(data: bytes) -> Optional[dict]:
    if len(data) < HEADER_SIZE:
        return None
    magic, version, msg_type, flags, payload_size, checksum, seq_id = struct.unpack(
        HEADER_FORMAT, data[:HEADER_SIZE])
    if magic != MAGIC_HEADER:
        return None
    return {
        "magic": magic, "version": version, "msg_type": msg_type,
        "flags": flags, "payload_size": payload_size,
        "checksum": checksum, "seq_id": seq_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Handshake
# ──────────────────────────────────────────────────────────────────────────────

def build_handshake_ack() -> bytes:
    """Resposta ao handshake UE5. Payload vazio — UE5 valida só cabeçalho."""
    return _build_frame(MSG_HANDSHAKE_ACK, 0, b"")


def parse_handshake(data: bytes) -> Optional[Tuple[int, int]]:
    """
    Retorna (obs_dim, emb_dim) ou None.
    C++ BuildHandshake envia: obs_dim(>I) + emb_dim(>I) + version(>I).
    """
    hdr = _parse_header(data)
    if hdr is None or hdr["msg_type"] != MSG_HANDSHAKE:
        return None
    payload = data[HEADER_SIZE:]
    if len(payload) < 12:
        return None
    obs_dim, emb_dim, ver = struct.unpack(">III", payload[:12])
    return obs_dim, emb_dim


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pack_vector3(buf: bytearray, v: np.ndarray) -> None:
    buf += struct.pack("<fff", float(v[0]), float(v[1]), float(v[2]))


def _pack_quat(buf: bytearray, q: np.ndarray) -> None:
    buf += struct.pack("<ffff", float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _pack_trajectory(buf: bytearray, traj: Trajectory) -> None:
    """
    Serializa trajetória para o wire format UE5.
    60 bytes por sample: Position(12) + LinearVelocity(12) + AngularVelocity(12)
                         + Facing(16) + TimeInSeconds(4) + Speed(4)
    Contagem de samples em big-endian (lida por ReadInt32 = BYTESWAP).
    """
    buf += struct.pack(">I", len(traj.samples))
    for s in traj.samples:
        _pack_vector3(buf, s.position)
        _pack_vector3(buf, s.linear_velocity)
        _pack_vector3(buf, s.angular_velocity)
        _pack_quat(buf, s.facing)
        buf += struct.pack("<ff", float(s.time_in_seconds), float(s.speed))


# ──────────────────────────────────────────────────────────────────────────────
# MSG_MOTION_RESPONSE (0x02)
# ──────────────────────────────────────────────────────────────────────────────

# Estados físicos → int (espelhado em UE5 ECognitivePhysicalState)
PHYSICAL_STATE_MAP = {
    "alive": 0, "dead": 1, "falling": 2, "swimming": 3, "landing": 4,
    "attack": 5, "flee": 6, "hide": 7, "pickup": 8, "enter": 9,
}

def _mat4_to_loc_rot_scale(mat: "np.ndarray"):
    """
    Decompõe uma matriz 4×4 (gerada por parse_pose_frame) em
    localização, quaternion XYZW e escala.
    """
    loc = mat[:3, 3].tolist()
    sx = float(np.linalg.norm(mat[:3, 0]))
    sy = float(np.linalg.norm(mat[:3, 1]))
    sz = float(np.linalg.norm(mat[:3, 2]))
    eps = 1e-8
    R = mat[:3, :3].copy()
    R[:, 0] /= max(sx, eps)
    R[:, 1] /= max(sy, eps)
    R[:, 2] /= max(sz, eps)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        w = 0.25 / s; x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]))
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]))
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * float(np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]))
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    return loc, [float(x), float(y), float(z), float(w)], [sx, sy, sz]


def build_motion_response(
    seq_id:             int,
    embedding:          np.ndarray,
    confidence:         float,
    refined_trajectory: Trajectory,
    selected_style:     int,
    latency_ms:         float,
    valid:              bool,
    bone_transforms:    Optional[list] = None,
    physical_state:     str = "alive",
) -> bytes:
    """
    Constrói MSG_MOTION_RESPONSE (0x02) aceito por UE5 DeserializeResponse().

    Layout:
      SequenceId    <q   LE int64
      SelectedStyle >I   BE uint32
      LatencyMs     <f   LE float
      bValid        >I   BE uint32
      Confidence    <f   LE float
      EmbedCount    >I + EmbedValues N×<f
      TrajCount     >I + Samples N×(Vec+Vec+Vec+Quat+f+f)
      BoneCount     >I + Bones N×(Loc12+Rot16+Scale12) = N×40 bytes
    """
    buf = bytearray()
    buf += struct.pack("<q", seq_id)
    buf += struct.pack(">I", int(selected_style))
    buf += struct.pack("<f", float(latency_ms))
    buf += struct.pack(">I", 1 if valid else 0)
    buf += struct.pack("<f", float(confidence))

    emb_arr = embedding.astype(np.float32).flatten()
    buf += struct.pack(">I", len(emb_arr))
    buf += emb_arr.tobytes()

    _pack_trajectory(buf, refined_trajectory)

    # Bone transforms: todos os bones do skeleton do NPC gerados pelo Python
    bones = bone_transforms if bone_transforms else []
    buf += struct.pack(">I", len(bones))
    for bone in bones:
        if isinstance(bone, np.ndarray) and bone.ndim == 2 and bone.shape == (4, 4):
            # Formato vindo de parse_pose_frame: matriz 4×4 — decompõe em loc/rot/scale
            loc, rot, scl = _mat4_to_loc_rot_scale(bone)
        elif isinstance(bone, dict):
            loc = bone.get("location", [0.0, 0.0, 0.0])
            rot = bone.get("rotation", [0.0, 0.0, 0.0, 1.0])
            scl = bone.get("scale",    [1.0, 1.0, 1.0])
        else:
            loc, rot, scl = [0.0]*3, [0.0, 0.0, 0.0, 1.0], [1.0]*3

        buf += struct.pack("<fff",  float(loc[0]), float(loc[1]), float(loc[2]))
        buf += struct.pack("<ffff", float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        buf += struct.pack("<fff",  float(scl[0]), float(scl[1]), float(scl[2]))

    # PhysicalState (>I BE) — UE5 lê após os bones; clientes antigos ignoram.
    buf += struct.pack(">I", PHYSICAL_STATE_MAP.get(physical_state, 0))

    return _build_frame(MSG_MOTION_RESPONSE, seq_id, bytes(buf))


# ──────────────────────────────────────────────────────────────────────────────
# MSG_MOTION_REQUEST parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_motion_request(data: bytes) -> Optional[dict]:
    hdr = _parse_header(data)
    if hdr is None or hdr["msg_type"] != MSG_MOTION_REQUEST:
        return None

    payload = data[HEADER_SIZE:]
    off = 0

    seq_id          = struct.unpack_from("<q", payload, off)[0]; off += 8
    requested_style = struct.unpack_from(">I", payload, off)[0]; off += 4
    max_latency_ms  = struct.unpack_from("<f", payload, off)[0]; off += 4

    pose_data_size   = struct.unpack_from(">I", payload, off)[0]; off += 4
    raw_pose_payload = payload[off:off + pose_data_size];          off += pose_data_size

    pose_frame = parse_pose_frame(bytes(raw_pose_payload))

    def read_trajectory(p: bytes, o: int) -> Tuple[Trajectory, int]:
        n = struct.unpack_from(">I", p, o)[0]; o += 4
        samples = []
        for _ in range(n):
            pos  = np.array(struct.unpack_from("<fff",  p, o), dtype=np.float32); o += 12
            linv = np.array(struct.unpack_from("<fff",  p, o), dtype=np.float32); o += 12
            angv = np.array(struct.unpack_from("<fff",  p, o), dtype=np.float32); o += 12
            fac  = np.array(struct.unpack_from("<ffff", p, o), dtype=np.float32); o += 16
            t, sp = struct.unpack_from("<ff", p, o); o += 8
            samples.append(TrajectorySample(
                position=pos, linear_velocity=linv, angular_velocity=angv,
                facing=fac, time_in_seconds=float(t), speed=float(sp),
            ))
        return Trajectory(samples=samples), o

    desired_traj, off = read_trajectory(payload, off)

    health          = struct.unpack_from("<f", payload, off)[0]; off += 4
    stamina         = struct.unpack_from("<f", payload, off)[0]; off += 4
    alertness       = struct.unpack_from("<f", payload, off)[0]; off += 4
    fear_level      = struct.unpack_from("<f", payload, off)[0]; off += 4
    aggression      = struct.unpack_from("<f", payload, off)[0]; off += 4
    threat_level    = struct.unpack_from("<f", payload, off)[0]; off += 4
    current_state   = struct.unpack_from(">I", payload, off)[0]; off += 4
    emotional_state = struct.unpack_from(">I", payload, off)[0]; off += 4

    return {
        "seq_id":             seq_id,
        "requested_style":    requested_style,
        "max_latency_ms":     max_latency_ms,
        "pose_frame":         pose_frame,
        "desired_trajectory": desired_traj,
        "blackboard": {
            "health":           health,
            "stamina":          stamina,
            "alertness":        alertness,
            "fear_level":       fear_level,
            "aggression_level": aggression,
            "threat_level":     threat_level,
            "current_state":    current_state,
            "emotional_state":  emotional_state,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# MSG_POSE_FRAME parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_pose_frame(data: bytes) -> Optional[PoseFrame]:
    if not data or len(data) < HEADER_SIZE:
        return PoseFrame.zero()
    hdr = _parse_header(data)
    if hdr is None or hdr["msg_type"] != MSG_POSE_FRAME:
        return PoseFrame.zero()

    payload = data[HEADER_SIZE:]
    off = 0

    def read_f64():  nonlocal off; v = struct.unpack_from("<d",    payload, off)[0]; off += 8;  return v
    def read_i32():  nonlocal off; v = struct.unpack_from(">I",    payload, off)[0]; off += 4;  return v
    def read_f32():  nonlocal off; v = struct.unpack_from("<f",    payload, off)[0]; off += 4;  return v
    def read_vec3(): nonlocal off; v = np.array(struct.unpack_from("<fff",  payload, off), dtype=np.float32); off += 12; return v
    def read_quat(): nonlocal off; v = np.array(struct.unpack_from("<ffff", payload, off), dtype=np.float32); off += 16; return v

    timestamp   = read_f64()
    frame_index = read_i32()
    lin_vel     = read_vec3()
    ang_vel     = read_vec3()
    root_loc    = read_vec3()
    root_rot    = read_quat()

    n_bones = read_i32()
    bones   = []
    for _ in range(n_bones):
        loc = read_vec3()
        rot = read_quat()
        scl = read_vec3()
        x, y, z, w = rot[0], rot[1], rot[2], rot[3]
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = (1 - 2*(y*y + z*z)) * scl[0]
        mat[0, 1] = (2*(x*y - z*w))     * scl[1]
        mat[0, 2] = (2*(x*z + y*w))     * scl[2]
        mat[1, 0] = (2*(x*y + z*w))     * scl[0]
        mat[1, 1] = (1 - 2*(x*x + z*z)) * scl[1]
        mat[1, 2] = (2*(y*z - x*w))     * scl[2]
        mat[2, 0] = (2*(x*z - y*w))     * scl[0]
        mat[2, 1] = (2*(y*z + x*w))     * scl[1]
        mat[2, 2] = (1 - 2*(x*x + y*y)) * scl[2]
        mat[:3, 3] = loc
        bones.append(mat)

    def read_traj() -> Trajectory:
        n = read_i32()
        samples = []
        for _ in range(n):
            pos = read_vec3()
            lv  = read_vec3()
            av  = read_vec3()
            fac = read_quat()
            t   = read_f32()
            sp  = read_f32()
            samples.append(TrajectorySample(pos, lv, av, fac, t, sp))
        return Trajectory(samples=samples)

    past_traj   = read_traj()
    future_traj = read_traj()

    n_curves = read_i32()
    curves   = {}
    for _ in range(n_curves):
        klen = read_i32()
        key  = payload[off:off + klen].decode("utf-8", errors="ignore"); off += klen
        val  = read_f32()
        curves[key] = val

    n_tags = read_i32()
    tags   = []
    for _ in range(n_tags):
        tlen = read_i32()
        tag  = payload[off:off + tlen].decode("utf-8", errors="ignore"); off += tlen
        tags.append(tag)

    movement_mode = read_i32()
    motion_style  = read_i32()

    return PoseFrame(
        timestamp=timestamp, frame_index=frame_index,
        root_location=root_loc, root_rotation=root_rot,
        linear_velocity=lin_vel, angular_velocity=ang_vel,
        bone_transforms=bones, past_trajectory=past_traj,
        future_trajectory=future_traj, curve_values=curves,
        tags=tags, movement_mode=movement_mode, motion_style=motion_style,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NPCId helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_int64(data: bytes, offset: int) -> Tuple[int, int]:
    v = struct.unpack_from("<q", data, offset)[0]
    return v, offset + 8


def _write_int64(v: int) -> bytes:
    return struct.pack("<q", int(v))


# ──────────────────────────────────────────────────────────────────────────────
# MSG_LEADER_SEQUENCE
#
# CORREÇÃO CP-01 + CP-02:
# Wire layout C++ SerializeLeaderSequence:
#   LeaderNPCId    (8 bytes, LE int64)      ← WriteInt64
#   FollowerNPCId  (8 bytes, LE int64)      ← WriteInt64
#   SequenceId     (4 bytes, BE uint32)     ← WriteInt32 (BYTESWAP_ORDER32)
#   StartTimestamp (8 bytes, LE double)     ← WriteDouble   ← ESTAVA FALTANDO
#   EndTimestamp   (8 bytes, LE double)     ← WriteDouble   ← ESTAVA FALTANDO
#   NumFrames      (4 bytes, BE uint32)     ← WriteInt32
#   [FrameSize(4 BE) + FrameData]*
#
# Total offset antes de NumFrames: 8+8+4+8+8 = 36 bytes (após header).
# Antes da correção Python pulava StartTs+EndTs e lia 4 bytes de StartTs como NumFrames.
# ──────────────────────────────────────────────────────────────────────────────

def build_leader_sequence(
    frames,
    leader_npc_id:   int,
    follower_npc_id: int,
    sequence_id:     int   = 0,
    start_timestamp: float = 0.0,
    end_timestamp:   float = 0.0,
) -> bytes:
    """
    Serializa MSG_LEADER_SEQUENCE compatível com C++ parse no lado Python
    e com C++ SerializeLeaderSequence no lado UE5.

    Layout:
      LeaderNPCId    (<q  LE int64)
      FollowerNPCId  (<q  LE int64)
      SequenceId     (>I  BE uint32)   ← CP-02 fix
      StartTimestamp (<d  LE double)   ← CP-01 fix (adicionado)
      EndTimestamp   (<d  LE double)   ← CP-01 fix (adicionado)
      NumFrames      (>I  BE uint32)
      [FrameSize(>I) + FrameData]*
    """
    payload = bytearray()
    payload += _write_int64(leader_npc_id)
    payload += _write_int64(follower_npc_id)
    payload += struct.pack(">I", int(sequence_id))          # CP-02: era "<i"
    payload += struct.pack("<d", float(start_timestamp))    # CP-01: estava ausente
    payload += struct.pack("<d", float(end_timestamp))      # CP-01: estava ausente
    payload += struct.pack(">I", len(frames))
    for frame_bytes in frames:
        payload += struct.pack(">I", len(frame_bytes))
        payload += frame_bytes
    return _build_frame(MSG_LEADER_SEQUENCE, sequence_id, bytes(payload))


def parse_leader_sequence(data: bytes) -> dict:
    """
    Desserializa MSG_LEADER_SEQUENCE vindo do UE5.

    CORREÇÃO CP-01: lê StartTimestamp + EndTimestamp antes de NumFrames.
    CORREÇÃO CP-02: SequenceId lido como ">I" (BE uint32), não "<i" (LE int32).
    """
    hdr = _parse_header(data)
    if hdr is None:
        return {}
    off = HEADER_SIZE

    # LeaderNPCId (8 LE)
    leader_npc_id,   off = _read_int64(data, off)
    # FollowerNPCId (8 LE)
    follower_npc_id, off = _read_int64(data, off)

    # CP-02: SequenceId (4 BE) — WriteInt32 usa BYTESWAP_ORDER32
    if off + 4 > len(data):
        return {}
    sequence_id = struct.unpack_from(">I", data, off)[0]; off += 4

    # CP-01: StartTimestamp (8 LE) — WriteDouble usa Memcpy (LE nativo)
    if off + 8 > len(data):
        return {}
    start_timestamp = struct.unpack_from("<d", data, off)[0]; off += 8

    # CP-01: EndTimestamp (8 LE)
    if off + 8 > len(data):
        return {}
    end_timestamp = struct.unpack_from("<d", data, off)[0]; off += 8

    # NumFrames (4 BE)
    if off + 4 > len(data):
        return {}
    n_frames = struct.unpack_from(">I", data, off)[0]; off += 4

    raw_frames = []
    for _ in range(n_frames):
        if off + 4 > len(data):
            break
        frame_size = struct.unpack_from(">I", data, off)[0]; off += 4
        if off + frame_size > len(data):
            break
        raw_frames.append(data[off:off + frame_size])
        off += frame_size

    return {
        "leader_npc_id":   int(leader_npc_id),
        "follower_npc_id": int(follower_npc_id),
        "sequence_id":     int(sequence_id),
        "start_timestamp": float(start_timestamp),
        "end_timestamp":   float(end_timestamp),
        "raw_frames":      raw_frames,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MSG_AUTONOMOUS_REQUEST
# ──────────────────────────────────────────────────────────────────────────────

def parse_autonomous_request(data: bytes) -> dict:
    hdr = _parse_header(data)
    if hdr is None:
        return {}
    off = HEADER_SIZE
    npc_id,        off = _read_int64(data, off)
    target_npc_id, off = _read_int64(data, off)

    inner_payload = data[off:]
    inner_size    = len(inner_payload)
    fixed_header  = struct.pack(
        HEADER_FORMAT,
        MAGIC_HEADER, PROTOCOL_VERSION, MSG_MOTION_REQUEST,
        0, inner_size, 0, hdr["seq_id"],
    )
    req = parse_motion_request(fixed_header + inner_payload)
    if req:
        req["npc_id"]        = int(npc_id)
        req["target_npc_id"] = int(target_npc_id)
    return req or {}


def build_autonomous_request(
    npc_id: int, target_npc_id: int,
    motion_request_payload: bytes, seq_id: int = 0,
) -> bytes:
    inner = _write_int64(npc_id) + _write_int64(target_npc_id) + motion_request_payload
    return _build_frame(MSG_AUTONOMOUS_REQUEST, seq_id, inner)


# ──────────────────────────────────────────────────────────────────────────────
# MSG_MOTION_ACTION
# ──────────────────────────────────────────────────────────────────────────────

def build_motion_action(
    seq_id:     int,
    npc_id:     int,
    action_idx: int,
    direction:  list,
    speed:      float,
    confidence: float = 1.0,
    latency_ms: float = 0.0,
    valid:      bool  = True,
) -> bytes:
    payload  = struct.pack("<q", seq_id)
    payload += _write_int64(npc_id)
    payload += struct.pack("<i", action_idx)
    payload += struct.pack("<fff", float(direction[0]), float(direction[1]), float(direction[2]))
    payload += struct.pack("<ff",  float(speed), float(confidence))
    payload += struct.pack("<f",   float(latency_ms))
    payload += struct.pack("<i",   1 if valid else 0)
    return _build_frame(MSG_MOTION_ACTION, seq_id, payload)


def parse_motion_action(data: bytes) -> dict:
    hdr = _parse_header(data)
    if hdr is None:
        return {}
    off = HEADER_SIZE
    seq_id     = struct.unpack_from("<q", data, off)[0]; off += 8
    npc_id     = struct.unpack_from("<q", data, off)[0]; off += 8
    action_idx = struct.unpack_from("<i", data, off)[0]; off += 4
    dx, dy, dz = struct.unpack_from("<fff", data, off);  off += 12
    speed, conf = struct.unpack_from("<ff", data, off);  off += 8
    latency_ms  = struct.unpack_from("<f",  data, off)[0]; off += 4
    valid       = struct.unpack_from("<i",  data, off)[0] != 0
    return {
        "seq_id": seq_id, "npc_id": npc_id, "action_idx": action_idx,
        "direction": [dx, dy, dz], "speed": speed,
        "confidence": conf, "latency_ms": latency_ms, "valid": valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MSG_PERCEPTION (0x08) — NPC envia entidades percebidas no mundo.
# Espelha SerializePerception em CognitiveMotionProtocol.cpp.
# Wire: NPCId(8 <q) + NumEntities(4 >I)
#       + [ Category(4 >I) Disposition(4 >I) Reaction(4 >I) VehicleType(4 >I)
#           TrafficState(4 >I) Distance(4 <f) DirX(4 <f) DirY(4 <f) DirZ(4 <f)
#           ThreatWeight(4 <f) ]*
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_CATEGORY_NAMES = {
    0: "unknown", 1: "character", 2: "weapon", 3: "pickup", 4: "vehicle",
    5: "traffic_light", 6: "cover", 7: "hazard", 8: "objective", 9: "ignore",
}
DISPOSITION_NAMES = {0: "neutral", 1: "friend", 2: "enemy", 3: "ally"}
ROLE_NAMES = {0: "none", 1: "hostage", 2: "captor", 3: "civilian",
              4: "wounded", 5: "leader"}
# Conjunto canônico de emoções (deve bater com EMOTIONS em demonstration_learning).
EMOTION_NAMES = {0: "calm", 1: "happy", 2: "alert", 3: "fear", 4: "anger",
                 5: "panic", 6: "confident", 7: "suspicious"}
REACTION_NAMES = {
    0: "none", 1: "approach", 2: "attack", 3: "flee", 4: "hide",
    5: "pickup", 6: "enter", 7: "wait", 8: "cross",
}


def parse_perception(data: bytes) -> dict:
    """Desserializa MSG_PERCEPTION. Retorna {} em erro."""
    hdr = _parse_header(data)
    if hdr is None:
        return {}
    off = HEADER_SIZE
    if off + 12 > len(data):
        return {}
    npc_id = struct.unpack_from("<q", data, off)[0]; off += 8
    n_ent  = struct.unpack_from(">I", data, off)[0]; off += 4

    entities = []
    for _ in range(n_ent):
        if off + 44 > len(data):
            break
        cat   = struct.unpack_from(">I", data, off)[0]; off += 4
        disp  = struct.unpack_from(">I", data, off)[0]; off += 4
        role  = struct.unpack_from(">I", data, off)[0]; off += 4
        react = struct.unpack_from(">I", data, off)[0]; off += 4
        veh   = struct.unpack_from(">I", data, off)[0]; off += 4
        traf  = struct.unpack_from(">I", data, off)[0]; off += 4
        dist  = struct.unpack_from("<f", data, off)[0]; off += 4
        dx    = struct.unpack_from("<f", data, off)[0]; off += 4
        dy    = struct.unpack_from("<f", data, off)[0]; off += 4
        dz    = struct.unpack_from("<f", data, off)[0]; off += 4
        threat= struct.unpack_from("<f", data, off)[0]; off += 4
        entities.append({
            "category": cat, "category_name": ENTITY_CATEGORY_NAMES.get(cat, "unknown"),
            "disposition": disp, "disposition_name": DISPOSITION_NAMES.get(disp, "neutral"),
            "role": role, "role_name": ROLE_NAMES.get(role, "none"),
            "reaction": react, "reaction_name": REACTION_NAMES.get(react, "none"),
            "vehicle_type": veh, "traffic_state": traf,
            "distance": dist, "direction": [dx, dy, dz], "threat_weight": threat,
        })

    return {"npc_id": npc_id, "entities": entities}


# ─────────────────────────────────────────────────────────────────────────────
# MSG_TEACH (0x09) — líder ensina o vocabulário de ações.
# Espelha SerializeTeach em CognitiveMotionProtocol.cpp.
# Wire: LeaderNPCId(8 <q) + CurrentVerb(4 >I) + LeaderCategory(4 >I)
#       + NumActions(4 >I)
#       + [ Verb(4 >I) ActionIndex(4 >I = signed via two's complement)
#           TargetCategory(4 >I) LabelLen(4 >I) LabelBytes ]*
# ─────────────────────────────────────────────────────────────────────────────
VERB_NAMES = {
    0: "idle", 1: "walk", 2: "run", 3: "jump", 4: "crouch", 5: "crawl",
    6: "vault", 7: "pickup", 8: "flee", 9: "hide", 10: "attack", 11: "defend",
}


def parse_teach(data: bytes) -> dict:
    """Desserializa MSG_TEACH. Retorna {} em erro."""
    hdr = _parse_header(data)
    if hdr is None:
        return {}
    off = HEADER_SIZE
    if off + 20 > len(data):
        return {}
    leader_id = struct.unpack_from("<q", data, off)[0]; off += 8
    cur_verb  = struct.unpack_from(">I", data, off)[0]; off += 4
    leader_cat= struct.unpack_from(">I", data, off)[0]; off += 4
    n_acts    = struct.unpack_from(">I", data, off)[0]; off += 4

    vocab = []
    for _ in range(n_acts):
        if off + 16 > len(data):
            break
        verb    = struct.unpack_from(">I", data, off)[0]; off += 4
        # ActionIndex pode ser -1 (verbo sem efetuador). WriteInt32 grava o
        # padrão de bits; reinterpretamos como signed.
        act_raw = struct.unpack_from(">I", data, off)[0]; off += 4
        act_idx = act_raw - 0x100000000 if act_raw >= 0x80000000 else act_raw
        tgt_cat = struct.unpack_from(">I", data, off)[0]; off += 4
        llen    = struct.unpack_from(">I", data, off)[0]; off += 4
        if off + llen > len(data):
            break
        label = data[off:off + llen].decode("utf-8", errors="ignore"); off += llen
        vocab.append({
            "verb": verb, "verb_name": VERB_NAMES.get(verb, "unknown"),
            "action_index": act_idx, "target_category": tgt_cat,
            "target_category_name": ENTITY_CATEGORY_NAMES.get(tgt_cat, "unknown"),
            "label": label,
        })

    # Rótulos de demonstração (opcionais, retrocompatível): emoção e ação que
    # o líder está demonstrando AGORA. Se os bytes não vierem (C++ antigo),
    # ficam vazios e o aprendizado por demonstração simplesmente não dispara.
    cur_emotion_idx = -1
    cur_action_idx  = -1
    if off + 8 <= len(data):
        cur_emotion_idx = struct.unpack_from(">I", data, off)[0]; off += 4
        a_raw = struct.unpack_from(">I", data, off)[0]; off += 4
        cur_action_idx = a_raw - 0x100000000 if a_raw >= 0x80000000 else a_raw

    return {
        "leader_npc_id": leader_id,
        "current_verb": cur_verb,
        "current_verb_name": VERB_NAMES.get(cur_verb, "unknown"),
        "leader_category": leader_cat,
        "vocabulary": vocab,
        "current_emotion_name": EMOTION_NAMES.get(cur_emotion_idx, ""),
        "current_action_index": cur_action_idx,
    }

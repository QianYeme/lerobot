import json
from pathlib import Path

ROOT = Path(__file__).parent


def check_info_json():
    """从 info.json 读取特征定义 — 这是数据采集时自动记录的元数据。"""
    with open(ROOT / "meta/info.json") as f:
        info = json.load(f)

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]

    # 基线维度（修改前只有这些）
    baseline = {
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    }
    expanded_state = [n for n in state_names if n not in baseline]
    expanded_action = [n for n in action_names if n not in baseline]

    print("=== info.json（数据集元数据） ===")
    print(f"  observation.state 维度: {state_dim}")
    for i, name in enumerate(state_names):
        tag = " [扩展]" if name in expanded_state else ""
        print(f"    [{i}] {name}{tag}")

    print(f"\n  action 维度: {action_dim}")
    for i, name in enumerate(action_names):
        tag = " [扩展]" if name in expanded_action else ""
        print(f"    [{i}] {name}{tag}")

    if expanded_state or expanded_action:
        print("\n  检测到维度扩展:")
        for n in expanded_state:
            print(f"    + observation: {n}")
        for n in expanded_action:
            print(f"    + action: {n}")
    else:
        print("\n  未检测到维度扩展")

    return info, state_names


def check_parquet_direct():
    """直接从 parquet 读取数据验证。"""
    import pyarrow.parquet as pq

    parquet_path = ROOT / "data/file-000.parquet"
    if not parquet_path.exists():
        # 尝试 chunk 目录结构
        parquet_path = ROOT / "data/chunk-000/file-000.parquet"
    if not parquet_path.exists():
        print(f"\n=== Parquet 验证: 文件不存在 ({parquet_path}) ===")
        return

    try:
        pf = pq.ParquetFile(str(parquet_path))
        batch = next(pf.iter_batches(batch_size=3))
        df = batch.to_pandas()
        print("\n=== Parquet 数据验证（前 3 帧） ===")
        print(f"  列名: {list(df.columns)}")

        if "observation.state" in df.columns:
            state_vec = df["observation.state"].iloc[0]
            print(f"  observation.state 向量长度: {len(state_vec)}")
            for i in range(min(3, len(df))):
                print(f"    frame {i}: {df['observation.state'].iloc[i]}")

        if "action" in df.columns:
            action_vec = df["action"].iloc[0]
            print(f"\n  action 向量长度: {len(action_vec)}")
            for i in range(min(3, len(df))):
                print(f"    frame {i}: {df['action'].iloc[i]}")
    except Exception as e:
        print(f"\n=== Parquet 读取失败: {e} ===")
        print("（不影响结论，info.json 已是权威记录）")


if __name__ == "__main__":
    info, state_names = check_info_json()
    # parquet 文件可能有版本兼容问题，用 try 包裹
    try:
        check_parquet_direct()
    except Exception as e:
        print(f"\n=== Parquet 验证异常: {e} ===")

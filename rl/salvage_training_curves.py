import json
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"^\|\s*([A-Za-z_/]+)\s*\|\s*([-\deE.+]+)\s*\|\s*$")


def parse_log(path):
    rows, cur = [], {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("----"):
            if cur.get("total_timesteps") is not None:
                rows.append(cur)
            cur = {}
            continue
        m = BLOCK.match(line.rstrip())
        if m:
            key, val = m.group(1).strip().rstrip("/"), m.group(2)
            try:
                cur[key] = float(val)
            except ValueError:
                pass
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["total_timesteps"]):
        t = r["total_timesteps"]
        if t not in seen:
            seen.add(t)
            out.append(r)
    return out


def main(log_dir="rl/logs", out_path="rl/training_curves.json"):
    curves = {}
    for log in sorted(Path(log_dir).glob("*.out")):
        rows = parse_log(log)
        if not rows:
            continue
        curves[log.stem] = {
            "log": str(log), "n_points": len(rows), "rows": rows,
            "final_ep_rew_mean": rows[-1].get("ep_rew_mean"),
            "final_timesteps": rows[-1].get("total_timesteps"),
            "mean_fps": (sum(r.get("fps", 0) for r in rows) / len(rows)),
        }
        print(f"  {log.stem:34} {len(rows):3d} points, "
              f"final ep_rew_mean {rows[-1].get('ep_rew_mean')}, "
              f"{rows[-1].get('total_timesteps'):.0f} steps")
    Path(out_path).write_text(json.dumps(curves, indent=2))
    print(f"\nSaved {out_path}  ({len(curves)} runs)")
    return curves


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))

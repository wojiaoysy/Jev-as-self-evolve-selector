"""Sequential, restartable multi-seed experiment runner; no concurrent GPU jobs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from jev_evolve.curves import eval_protocol, load_dev, plot_suite
from jev_evolve.data import load_split
from jev_evolve.io import digest, load_torch, read_json, tensor_digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--data", default="data/gsm8k")
    p.add_argument("--adapter", default="runs/initial.pt")
    p.add_argument("--retention-dev")
    p.add_argument("--retention-data")
    p.add_argument("--dev", default="data/dev_fixed.json")
    p.add_argument("--output", default="runs/learning_curves")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--episodes", type=int)
    p.add_argument("--mode", choices=["full", "online-only"], default="full")
    p.add_argument("--with-pdq", action="store_true")
    p.add_argument("--ablations", action="store_true", help="Eight groups including forced Top-2 and PDQ D controls; defaults to 80 episodes")
    p.add_argument("--reuse-offline", help="Existing compatible full suite: reuse each seed's router.pt and pdq_offline.pt")
    p.add_argument("--pdq-mode", choices=["dynamic", "continual", "accumulate"], default="accumulate")
    p.add_argument("--pdq-accumulation-scale", type=float, default=1.0)
    p.add_argument("--update-mode", choices=["direct", "guarded"], default="direct")
    p.add_argument("--router", help="Required for online-only: a single existing router shared by all seeds")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.plot_only:
        plot_suite(a.output)
        return
    # Fail before an expensive run if plotting dependencies are not installed.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(a.output).resolve() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    if len(a.seeds) < 2 or len(set(a.seeds)) != len(a.seeds) or a.eval_every < 1:
        p.error("Need at least two distinct seeds and a positive evaluation interval")
    cfg = read_json(a.config)
    if a.ablations:
        a.with_pdq = True
        if cfg["router"]["top_k"] != 2 or cfg["adapter"]["K"] < 2:
            p.error("Ablations require random top_k=2 and at least two subspaces")
        if a.episodes is None:
            a.episodes = 80
    if a.with_pdq and (a.mode != "full" or a.update_mode != "direct"):
        p.error("Five-way PDQ comparison requires full mode and direct updates")
    data = Path(a.data).resolve()
    dev_path, initial_path = Path(a.dev).resolve(), Path(a.adapter).resolve()
    dev = load_dev(dev_path, data)
    initial = load_torch(initial_path)
    total = len(load_split(data, "online"))
    episodes = total if a.episodes is None else a.episodes
    if not 1 <= episodes <= total:
        p.error("episodes must be between 1 and the size of the online split")
    if a.mode == "online-only" and not a.router:
        p.error("online-only mode requires --router")
    shared = Path(a.router).resolve() if a.router else None
    router_hash = tensor_digest(load_torch(shared)["state_dict"].items()) if shared else None
    plan = {"format": 1, "seeds": a.seeds, "interval": a.eval_every, "episodes": episodes,
            "policies": ["base", "router", "random", "all"] + (["pdq"] if a.with_pdq else []) +
                        (["router_top2", "pdq_shuffle", "pdq_identity"] if a.ablations else []),
            "update_mode": a.update_mode, "pdq_mode": a.pdq_mode if a.with_pdq else None,
            "mode": a.mode, "config": cfg, "data_id": dev["parent_data_id"], "dev_id": dev["id"],
            "evaluation_protocol": eval_protocol(cfg),
            "initial_identity": initial["identity"], "initial_R": tensor_digest([("R", initial["adapter"]["R"])]),
            "shared_router_hash": router_hash}
    reuse = Path(a.reuse_offline).resolve() if a.reuse_offline else None
    if a.with_pdq and a.pdq_mode == "accumulate":
        import math
        if not math.isfinite(a.pdq_accumulation_scale) or a.pdq_accumulation_scale <= 0:
            p.error("PDQ accumulation scale must be finite and positive")
        plan["pdq_accumulation_scale"] = a.pdq_accumulation_scale
    if reuse:
        if a.mode != "full":
            p.error("reuse-offline requires full mode")
        previous = read_json(reuse / "suite.json")
        for key in ("config", "data_id", "initial_identity", "initial_R"):
            if previous[key] != plan[key]:
                raise ValueError(f"Reuse suite differs in {key}; use the original config/data/initial adapter")
        offline_hashes = {}
        for seed in a.seeds:
            directory = reuse / f"seed_{seed}"
            payload = load_torch(directory / "router.pt")
            offline_hashes[str(seed)] = {"router": tensor_digest(payload["state_dict"].items())}
            if a.with_pdq:
                payload = load_torch(directory / "pdq_offline.pt")
                offline_hashes[str(seed)]["pdq"] = tensor_digest([("R", payload["adapter"]["R"])])
        plan["reused_offline"] = {"path": str(reuse), "hashes": offline_hashes}
    retention_args = []
    if bool(a.retention_dev) != bool(a.retention_data):
        p.error("Supply both retention-dev and retention-data")
    if a.retention_dev:
        retention = load_dev(a.retention_dev, a.retention_data)
        if any(r.get("task", "gsm8k") != "gsm8k" for r in retention["rows"]):
            p.error("Retention dataset must be GSM8K")
        plan["retention"] = {"id": retention["id"], "protocol": eval_protocol(cfg, "gsm8k")}
        retention_args = ["--retention-dev", Path(a.retention_dev).resolve(), "--retention-data", Path(a.retention_data).resolve()]
    out = Path(a.output).resolve()
    plan_path = out / "suite.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Suite settings changed; use a new output folder")
    write_json(plan_path, plan)

    def run(arguments, log):
        command = [sys.executable, "-u", "-m", "jev_evolve.cli", *map(str, arguments)]
        print("RUN:", " ".join(command), flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\nCOMMAND " + json.dumps(command) + "\n")
            stream.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait():
                    raise RuntimeError(f"Command failed. See {log}. Rerun the identical suite command to resume.")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()

    for seed in a.seeds:
        folder = out / f"seed_{seed}"
        seed_cfg = dict(cfg)
        if a.mode == "full":
            seed_cfg["seed"] = seed
        config = folder / "config.json"
        write_json(config, seed_cfg)
        common = ["--config", config, "--data", data, "--adapter", initial_path]
        router = shared
        if a.mode == "full" and not reuse:
            records = folder / "phase1"
            run(["collect", *common, "--output", records], folder / "collect.log")
            router = folder / "router.pt"
            if not router.exists():
                run(["train-router", "--config", config, "--records", records, "--output", router], folder / "router_train.log")
        if reuse:
            router = reuse / f"seed_{seed}" / "router.pt"
        pdq = (reuse / f"seed_{seed}" if reuse else folder) / "pdq_offline.pt"
        if a.with_pdq and not pdq.exists() and not reuse:
            run(["train-pdq", *common, "--records", folder / "phase1", "--output", pdq], folder / "pdq_train.log")
        for policy in plan["policies"]:
            extra = ["--router", router] if policy in ("router", "router_top2") else []
            if policy.startswith("pdq"):
                extra = ["--pdq", pdq, "--pdq-mode", a.pdq_mode]
                if a.pdq_mode == "accumulate":
                    extra += ["--pdq-accumulation-scale", str(a.pdq_accumulation_scale)]
            run(["adapt", *common, "--policy", policy, "--output", folder / policy,
                 "--dev", dev_path, "--eval-every", a.eval_every, "--online-seed", seed,
                 "--max-episodes", episodes, "--update-mode", a.update_mode, *retention_args, *extra], folder / f"{policy}.log")
    plot_suite(out)
    print(f"DONE: {out / 'learning_curves.png'}", flush=True)


if __name__ == "__main__":
    main()

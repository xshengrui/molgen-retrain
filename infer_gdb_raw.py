import argparse
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from ase.neighborlist import natural_cutoffs, neighbor_list


ATOM_TO_ONEHOT = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
}
IDX_TO_SYMBOL = {v: k for k, v in ATOM_TO_ONEHOT.items()}
TO_DEVICE_KEYS = [
    "species",
    "x",
    "cell",
    "num_atoms",
    "mask",
    "v_mask",
    "TKS_mask",
    "TKS_v_mask",
    "fragments_idx",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TPS inference on raw GDB reaction tarballs."
    )
    parser.add_argument(
        "--ckpt",
        default="epoch=829-step=0131970-val_meanRMSD_Kabsch=0.1485.ckpt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--tar",
        nargs="+",
        default=["data/GDB-10-rxn_raw.tar.gz", "data/GDB-17-rxn_raw.tar.gz"],
        help="One or more raw GDB reaction tar.gz files.",
    )
    parser.add_argument(
        "--out-root",
        default="experiments/gdb_raw_inference",
        help="Root directory for inference outputs.",
    )
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument(
        "--trial-batch-size",
        type=int,
        default=1,
        help="Number of stochastic trials for one reaction to infer in one GPU batch.",
    )
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap per tarball, useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a rollout directory if all requested gentraj files already exist.",
    )
    return parser.parse_args()


def fragment_atoms_ase(atoms, mult=1.50, min_bond=0.40):
    n_atoms = len(atoms)
    if n_atoms == 0:
        return []

    cutoffs = natural_cutoffs(atoms, mult=mult)
    i_idx, j_idx, distances = neighbor_list("ijd", atoms, cutoffs)
    parent = list(range(n_atoms))
    rank = [0] * n_atoms

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a_idx, b_idx):
        root_a = find(a_idx)
        root_b = find(b_idx)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_b] < rank[root_a]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for a_idx, b_idx, distance in zip(i_idx, j_idx, distances):
        if distance >= min_bond:
            union(int(a_idx), int(b_idx))

    groups = {}
    for idx in range(n_atoms):
        groups.setdefault(find(idx), []).append(idx)
    fragments = list(groups.values())
    fragments.sort(key=lambda group: min(group))
    return fragments


def fragments_index(symbols, positions):
    atoms = Atoms(symbols=symbols, positions=positions)
    fragments = fragment_atoms_ase(atoms)
    frag_idx = np.zeros(len(symbols), dtype=np.int64)
    for idx, fragment in enumerate(fragments):
        frag_idx[fragment] = idx
    return frag_idx


def parse_xyz_text(text):
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        raise ValueError("empty xyz file")
    n_atoms = int(lines[0].strip())
    atom_lines = [line for line in lines[2:] if line.strip()][:n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"expected {n_atoms} atoms, found {len(atom_lines)}")

    symbols = []
    positions = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(positions, dtype=np.float64)


def one_hot_species(symbols):
    z = np.zeros((len(symbols), 5), dtype=np.float32)
    for atom_idx, symbol in enumerate(symbols):
        if symbol not in ATOM_TO_ONEHOT:
            raise ValueError(f"unsupported element {symbol!r}; expected H/C/N/O")
        z[atom_idx, ATOM_TO_ONEHOT[symbol]] = 1.0
    return z


def centered(positions):
    return positions - positions.mean(axis=0, keepdims=True)


def reaction_dirs(tar):
    dirs = {}
    for member in tar.getmembers():
        parts = member.name.split("/")
        if len(parts) >= 3 and parts[0] == "raw":
            dirs.setdefault("/".join(parts[:2]), set()).add(parts[2])
    required = {"R.xyz", "TS.xyz", "P.xyz"}
    return sorted(dirname for dirname, files in dirs.items() if required.issubset(files))


def read_member_text(tar, member_name):
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    return extracted.read().decode("utf-8", errors="replace")


def make_batch(tar, dirname):
    frames = {}
    for label, filename in [("R", "R.xyz"), ("TS", "TS.xyz"), ("P", "P.xyz")]:
        symbols, positions = parse_xyz_text(read_member_text(tar, f"{dirname}/{filename}"))
        frames[label] = (symbols, positions)

    symbols = frames["R"][0]
    if not (symbols == frames["TS"][0] == frames["P"][0]):
        raise ValueError(f"atom symbols/order differ across R/TS/P for {dirname}")

    species = torch.tensor(
        np.stack([one_hot_species(symbols)] * 3), dtype=torch.float32
    )
    x = torch.tensor(
        np.stack(
            [
                centered(frames["R"][1]),
                centered(frames["TS"][1]),
                centered(frames["P"][1]),
            ]
        ),
        dtype=torch.float32,
    )
    fragments_idx = torch.tensor(
        np.stack(
            [
                fragments_index(symbols, frames["R"][1]),
                fragments_index(symbols, frames["TS"][1]),
                fragments_index(symbols, frames["P"][1]),
            ]
        ),
        dtype=torch.long,
    )

    n_atoms = len(symbols)
    mask = torch.ones((3, n_atoms), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1) * 200.0
    return {
        "name": Path(dirname).name,
        "species": species,
        "x": x,
        "fragments_idx": fragments_idx,
        "num_atoms": torch.tensor([n_atoms, n_atoms, n_atoms], dtype=torch.long),
        "cell": cell,
        "mask": mask,
        "v_mask": mask.unsqueeze(-1).expand(-1, -1, 3),
        "h_mask": mask.unsqueeze(-1).expand(-1, -1, 5),
        "TKS_mask": mask.clone(),
        "TKS_v_mask": mask.unsqueeze(-1).expand(-1, -1, 3).clone(),
        "TKS_h_mask": mask.unsqueeze(-1).expand(-1, -1, 5).clone(),
        "e_now": torch.zeros(3, dtype=torch.float32),
    }


def collate_repeat(item, repeat):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).expand(repeat, *value.shape).clone()
        else:
            batch[key] = [value] * repeat
    return batch


def move_to_device(batch, device):
    for key in TO_DEVICE_KEYS:
        if key in batch:
            batch[key] = batch[key].to(device)
    if "e_now" in batch:
        batch["e_now"] = batch["e_now"].to(device)
    return batch


def load_model(ckpt_path, device, inference_steps):
    from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

    ckpt = torch.load(
        ckpt_path,
        map_location={"cuda:3": "cuda:0", "cuda:2": "cuda:0", "cuda:1": "cuda:0"},
        weights_only=False,
    )
    hparams = ckpt["hyper_parameters"]
    hparams["args"].guided = False
    hparams["args"].inference_steps = inference_steps
    model = EquivariantMDGenWrapper(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def rollout(model, batch):
    positions, _ = model.inference(batch)
    return positions


def write_xyz_outputs(out_dir, batch, pred_pos, trial_idx, write_ref, batch_idx=0):
    labels = (
        torch.argmax(batch["species"], dim=3)[batch_idx]
        .detach()
        .cpu()
        .numpy()
    )
    symbols_by_frame = [
        [IDX_TO_SYMBOL[int(elem)] for elem in labels[frame_idx]]
        for frame_idx in range(labels.shape[0])
    ]

    pred_pos_np = pred_pos[batch_idx].detach().cpu().numpy()
    ref_pos_np = batch["x"][batch_idx].detach().cpu().numpy()
    cell = np.eye(3) * 25.0

    gen_file = out_dir / f"gentraj_{trial_idx}.xyz"
    if gen_file.exists():
        gen_file.unlink()
    for frame_idx in range(pred_pos_np.shape[0]):
        atoms = Atoms(symbols=symbols_by_frame[frame_idx], positions=pred_pos_np[frame_idx])
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, True])
        write(str(gen_file), atoms, append=True)

    if write_ref:
        ref_file = out_dir / "reftraj_1.xyz"
        if ref_file.exists():
            ref_file.unlink()
        for frame_idx in range(ref_pos_np.shape[0]):
            atoms = Atoms(symbols=symbols_by_frame[frame_idx], positions=ref_pos_np[frame_idx])
            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])
            write(str(ref_file), atoms, append=True)


def safe_dataset_name(tar_path):
    name = Path(tar_path).name
    for suffix in [".tar.gz", ".tgz", ".gz"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(tar_path).stem


def run_tar(model, tar_path, out_root, device, args):
    dataset_name = safe_dataset_name(tar_path)
    dataset_out = Path(out_root) / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)

    errors = []
    mappings = []
    with tarfile.open(tar_path, "r:gz") as tar:
        dirs = reaction_dirs(tar)
        selected = dirs[args.start :]
        if args.max_items is not None:
            selected = selected[: args.max_items]

        for rollout_idx, dirname in enumerate(selected):
            global_idx = args.start + rollout_idx
            rollout_id = f"rollout_{global_idx}"
            rxn_name = Path(dirname).name
            rollout_dir = dataset_out / rollout_id
            expected = [rollout_dir / f"gentraj_{trial}.xyz" for trial in range(args.num_trials)]
            if args.skip_existing and all(path.exists() for path in expected):
                mappings.append((rollout_id, rxn_name))
                continue

            print(f"[{dataset_name}] {global_idx + 1}/{len(dirs)} {rxn_name}", flush=True)
            try:
                item = make_batch(tar, dirname)
            except Exception as exc:
                errors.append((rxn_name, "prepare", repr(exc)))
                print(f"WARNING: failed to prepare {rxn_name}: {exc}", flush=True)
                continue

            rollout_dir.mkdir(parents=True, exist_ok=True)
            mappings.append((rollout_id, rxn_name))
            (rollout_dir / "README.md").write_text(
                f"Raw dataset: {tar_path}\nReaction: {rxn_name}\n",
                encoding="utf-8",
            )
            np.savetxt(
                rollout_dir / "Fragment_idx.dat",
                item["fragments_idx"].detach().cpu().numpy(),
                fmt="%d",
            )

            for trial_start in range(0, args.num_trials, args.trial_batch_size):
                trial_end = min(trial_start + args.trial_batch_size, args.num_trials)
                current_batch_size = trial_end - trial_start
                batch = move_to_device(collate_repeat(item, current_batch_size), device)
                start_time = time.time()
                try:
                    pred_pos = rollout(model, batch)
                except Exception as exc:
                    errors.append((rxn_name, f"trials_{trial_start}_{trial_end - 1}", repr(exc)))
                    print(
                        f"WARNING: inference failed for {rxn_name} trials "
                        f"{trial_start}-{trial_end - 1}: {exc}",
                        flush=True,
                    )
                    continue

                for batch_idx, trial_idx in enumerate(range(trial_start, trial_end)):
                    try:
                        write_xyz_outputs(
                            rollout_dir,
                            batch,
                            pred_pos,
                            trial_idx=trial_idx,
                            write_ref=(trial_idx == 0),
                            batch_idx=batch_idx,
                        )
                    except Exception as exc:
                        errors.append((rxn_name, f"write_trial_{trial_idx}", repr(exc)))
                        print(
                            f"WARNING: writing failed for {rxn_name} trial {trial_idx}: {exc}",
                            flush=True,
                        )
                print(
                    f"  trials {trial_start}-{trial_end - 1} "
                    f"(batch={current_batch_size}) finished in {time.time() - start_time:.2f}s",
                    flush=True,
                )

    with (dataset_out / "rollout_to_rxn_mapping.txt").open("w", encoding="utf-8") as fp:
        for rollout_id, rxn_name in mappings:
            fp.write(f"{rollout_id}\t{rxn_name}\n")

    if errors:
        with (dataset_out / "failed_reactions.tsv").open("w", encoding="utf-8") as fp:
            fp.write("reaction\tstage\terror\n")
            for rxn_name, stage, error in errors:
                fp.write(f"{rxn_name}\t{stage}\t{error}\n")


def main():
    args = parse_args()
    if args.trial_batch_size < 1:
        raise ValueError("--trial-batch-size must be >= 1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "README.md").write_text(
        f"Checkpoint: {args.ckpt}\n"
        f"Inference steps: {args.inference_steps}\n"
        f"Trials per reaction: {args.num_trials}\n"
        f"Trial batch size: {args.trial_batch_size}\n",
        encoding="utf-8",
    )

    model = load_model(args.ckpt, device, args.inference_steps)
    for tar_path in args.tar:
        run_tar(model, tar_path, args.out_root, device, args)


if __name__ == "__main__":
    main()

import pickle
import struct
from pathlib import Path
from tqdm import tqdm


def unpack_faceset_pak(pak: Path = Path("faceset.pak"), sp: Path = Path("out")):

    sp.mkdir(exist_ok=True)

    with pak.open("rb") as f:
        (version,) = struct.unpack("Q", f.read(8))
        assert version == 1

        (cfg_len,) = struct.unpack("Q", f.read(8))
        samples_configs = pickle.loads(f.read(cfg_len))

        offsets = [struct.unpack("Q", f.read(8))[0] for _ in range(len(samples_configs) + 1)]

        data_start = f.tell()

        for i, cfg in enumerate(tqdm(samples_configs, desc="Unpacking samples", unit="file")):
            start = data_start + offsets[i]
            end = data_start + offsets[i + 1]

            f.seek(start)
            raw = f.read(end - start)

            out_path = sp / cfg["filename"]
            out_path.write_bytes(raw)


if __name__ == "__main__":
    unpack_faceset_pak()

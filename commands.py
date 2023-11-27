import sys

import fire
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

from tirechecktool.infer import infer
from tirechecktool.train import train
from tirechecktool.export import export_onnx
from tirechecktool.run_server import run_server

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export": export_onnx,
            "run_server": run_server,
        }
    )

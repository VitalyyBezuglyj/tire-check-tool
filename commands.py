import sys

import fire
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

from tirechecktool.export import export_onnx
from tirechecktool.infer import infer
from tirechecktool.run_server import run_server
from tirechecktool.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export": export_onnx,
            "run_server": run_server,
        }
    )

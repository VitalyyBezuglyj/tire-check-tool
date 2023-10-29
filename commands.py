import fire

from tirechecktool.infer import infer
from tirechecktool.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )

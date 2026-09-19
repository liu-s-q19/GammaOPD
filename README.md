# GammaOPD

This repository contains the `verl`-based implementation of GammaOPD accompanying the
paper [Beyond Token-Local Imitation: Reward-Compatible Temporal Credit
Assignment for On-Policy Distillation](https://arxiv.org/abs/2609.16937).

Install dependencies from `requirements.txt`, then set model and dataset paths
before launching:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export STUDENT_MODEL=/path/to/student
export TEACHER_MODEL=/path/to/teacher
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/validation.parquet

bash scripts/opd.sh       # standard OPD
bash scripts/gammaopd.sh  # GammaOPD alternative
```

Runtime paths, cluster settings, batch sizes, and output locations are
environment-variable overrides.

## Citation

```bibtex
@misc{liu2026beyond,
  title         = {Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation},
  author        = {Liu, Shiqi and He, Zeyu and Tao, Letian and Zhan, Guojian and Gao, Jiaxin and Zhang, Feihong and Duan, Jingliang and Xiong, Wei and Sheng, Kehua and Zhang, Bo and Guan, Yang and Li, Shengbo Eben},
  year          = {2026},
  eprint        = {2609.16937},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.16937}
}
```

## Acknowledgements

We thank the authors and contributors of [verl](https://github.com/volcengine/verl),
[G-OPD](https://github.com/RUCBM/G-OPD) for their open-source contributions.

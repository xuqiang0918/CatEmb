# CatEmb SDAA 适配文档

## 结论

CatEmb（催化剂嵌入描述符模型，输入配体 SMILES、输出立体电子感知的描述符向量）已在 SDAA 平台适配完成，**零代码改动**：官方用法 `CatEmb(device='sdaa')` 直传即跑通。验收用官方推理锚点（作者预计算 npy）与官方 QSPR benchmark 两层闭环，精度均为 float32 噪声级一致。

## 环境（SDAA 侧）

| 项     | 值                                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------------- |
| 机器   | 10.71.15.115 容器 `sdaa`（镜像 3.2.1-torch_sdaa3.2.1）                                                  |
| Python | 3.12.13（/home/py312/bin/python3）                                                                      |
| 依赖   | torch_geometric 2.8.0 / torch_scatter 2.1.2 / e3nn 0.5.1 / rdkit 2026.03.5（容器内现成）                |
| 权重   | `catemb/model_path/dim32LN/`（best_model.pt + full_params.npy，modelscope `XuLiCheng2025/CatEmb` 下载） |
| 数据   | `dataset/`（figshare CatCompDB 数据包 `dataset.tar.gz` 解压，md5 `6d4a79f8fe043fa24f34732ea4b3f26a`）   |

## 改动

无。源码零改动、无 patches/ 目录：官方入口 `CatEmb(device=...)` 本来就把设备参数化了，`'sdaa'` 直传即用。

## 运行

官方推理用法（描述符生成，任意配体 SMILES 列表）：

```python
from catemb import CatEmb
calc = CatEmb(device="sdaa")
desc = calc.gen_desc(["CN(C)c1ccc(P(C2CCCCC2)C2CCCCC2)cc1"])   # 返回 (n, 32) ndarray
```

- 输入就是 `gen_desc()` 的 SMILES 列表参数，换数据集 = 换传入的 SMILES；无其他数据路径要配
- 权重默认从 `catemb/model_path/dim32LN/` 读取（换维度传 `model_path=` 指向对应 `dim*LN/`）
- `dataset/` 仅 QSPR benchmark 脚本需要（`--dataset thiol` 读 `dataset/rxn_data/NS_acetal_dataset_with_pdt.csv`）

## 快速验证

脚本在仓库根目录下执行（设备为位置参数 `sdaa|cuda|cpu`）：

| 脚本                             | 用途                                                                                                | 命令                                         | 产出                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `sdaa/scripts/official_infer.py` | 官方锚点：1536 条 aryl-scope 配体重推理（含计时、峰值显存），对照 `gen_desc/aryl_scope_cat_emb.npy` | `python sdaa/scripts/official_infer.py sdaa` | 逐点差 / `gen_s` / `peak_mem_gib`（stdout）+ `/tmp/catemb_aryl_<device>.npy` |

预期结果：`max ≤ 1e-6`、`nan=0`；sdaa 下 `gen_s≈11`、`peak_mem_gib≈0.21`。

官方 QSPR benchmark（数据齐全的 thiol 数据集，命令即官方验收入口）：

```bash
cd scripts && python qspr_catemb_all.py --dataset thiol --mode both --catemb-dims 32
```

产出 `benchmark_results/catemb_all_qspr/` 下 random / oos 两张 summary csv + qspr json，约 24 s。

## 性能

| 测例                     | 规模         | CUDA（1×A100，device=cuda） | SDAA（单卡） | CPU（同容器） | 倍数（CUDA/SDAA） |
| ------------------------ | ------------ | --------------------------- | ------------ | ------------- | ----------------- |
| 1536 条官方配体 gen_desc | 1536 × 32 维 | 3.71 s                      | 11.13 s      | 11.50 s       | ≈3.0×             |

峰值显存：SDAA 侧 0.207 GiB，CUDA 侧 0.187 GiB，同量级。慢于 CUDA 的原因是小图多 kernel 的发射开销主导，非算力瓶颈（SDAA 与 CPU 基本持平）。

## 精度

官方锚点：`gen_desc/aryl_scope_cat_emb.npy` 为作者预计算的 1536 条 aryl-scope 配体描述符（1536×32）。同一批 SMILES 在各平台重推理后逐点对照：

| 对照             | max 逐点差    | 100% < 1e-5 | NaN |
| ---------------- | ------------- | ----------- | --- |
| SDAA vs 官方 npy | 8.0e-07       | 是          | 0   |
| CPU vs 官方 npy  | 0（逐位一致） | 是          | 0   |
| CUDA vs 官方 npy | 8.3e-07       | 是          | 0   |
| SDAA vs CUDA     | 6.6e-07       | 是（<1e-6） | 0   |

CPU 与官方预计算逐位一致，说明官方 npy 为 CPU 产物，SDAA/CUDA 与它的差异即 float32 噪声级，无平台系统性偏差。

官方 QSPR benchmark（`scripts/qspr_catemb_all.py --dataset thiol --mode both --catemb-dims 32`，10 seeds，thiol ΔΔG 预测，1075 条反应，SDAA 侧端到端 24 s）。切分均为官方默认：random = 随机 475/1075 ≈ 44% 做测试 × 10 seeds 取均值（催化剂/底物训练时都见过，考插值）；三种 OOS = 作者在 `scripts/oos_splits.py` 固定的外推切分（催化剂/底物未在训练集出现，条数见表头）。SDAA（115）与 CUDA 机器（10.10.6.21，同一命令）双侧对照，表中每格 `SDAA r² / CUDA 机 r²`：

| 模型        | random          | catalyst_oos（304） | substrate_oos（216） | substrate_catalyst_oos（171） |
| ----------- | --------------- | ------------------- | -------------------- | ----------------------------- |
| ExtraTrees  | 0.8950 / 0.8950 | 0.3560 / 0.3560     | 0.8629 / 0.8629      | 0.2634 / 0.2634               |
| Ridge       | 0.8326 / 0.8326 | 0.1552 / 0.1553     | -0.1756 / -0.1747    | -0.2977 / -0.2968             |
| SVR         | 0.7913 / 0.7913 | 0.3111 / 0.3111     | -2.4801 / -2.4808    | -3.0934 / -3.0941             |
| KernelRidge | 0.6032 / 0.6032 | 0.3611 / 0.3611     | -1.1877 / -1.1877    | -1.7716 / -1.7716             |

random 划分 ExtraTrees r²=0.895（MAE=0.152），与论文报告量级一致；外推划分线性/核模型转负为该 benchmark 固有形态。两侧主要指标逐位一致（random ExtraTrees r² 在两侧均为 0.8949985793062118），个别项 1e-4 内尾差来自 sklearn 版本差（1.9.0 vs 1.7.2）。该 benchmark 描述符生成官方写死 `device="cpu"`、回归为 sklearn，计算不随加速器变化。

已知限制：aryl-scope 数据集无法复跑——脚本依赖的 `dataset/rxn_data/ligand_catalyst_map_of_aryl.csv` 官方未随数据包发布（figshare tar 与 modelscope 均无），非适配问题。

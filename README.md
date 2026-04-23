# 基于跨平台语义通信的神经网络部署方法研究（Python + PyTorch 工程）

本项目实现了一个可运行、可扩展的研究型原型系统，面向**图像语义通信**任务，支持：
- 语义编解码训练与评估（分类 + 重建双任务）
- 信道扰动模拟（AWGN / 丢包 / 带宽受限 / mixed）
- 边缘-云协同任务切分与本地双进程模拟
- TorchScript 导出、动态量化、一致性检查、benchmark
- 实验日志、曲线图、CSV/JSON 结果保存

---

## 1. 功能列表

1. 数据模块：CIFAR-10 / MNIST 下载、增强、训练/验证/测试划分、语义特征导出。
2. 模型模块：轻量 CNN 编码器 + 信道模拟 + 解码器 + 分类头，支持 p0/p1/p2 切分点。
3. 训练评估模块：train/eval/test 脚本、loss/metrics/evaluator、best/last ckpt。
4. 部署优化模块：TorchScript 导出、动态量化（默认）、一致性检查、性能 benchmark。
5. 边缘-云协同模块：socket 通信协议、cloud 服务、edge 客户端、demo 一键启动。
6. 可视化模块：训练曲线（loss/acc/psnr）、history.csv/json。

---

## 2. 目录结构

```text
project_root/
  README.md
  requirements.txt
  configs/
    default.yaml
    cifar10_train.yaml
    mnist_quick.yaml
    deploy_edge.yaml
    deploy_cloud.yaml
  data/
  datasets/
    __init__.py
    transforms.py
    cifar10_dataset.py
    mnist_dataset.py
    feature_export.py
  models/
    __init__.py
    blocks.py
    semantic_encoder.py
    semantic_decoder.py
    channel.py
    heads.py
    semantic_comm_model.py
    partition.py
  training/
    trainer.py
    losses.py
    metrics.py
    evaluator.py
  deployment/
    export_torchscript.py
    quantize.py
    consistency_check.py
    benchmark.py
    device_profile.py
  services/
    edge_server.py
    cloud_server.py
    protocol.py
    client_demo.py
  scripts/
    train.py
    eval.py
    test.py
    run_edge.py
    run_cloud.py
    run_demo.py
    run_benchmark.py
  utils/
    seed.py
    logger.py
    io.py
    visualization.py
    config.py
  outputs/
  tests/
    test_model_shapes.py
    test_partition.py
    test_export.py
```

---

## 3. 环境安装（Windows 11 / Python 3.11 / PyTorch 2.9.1）

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 如果你已经安装了 PyTorch 2.9.1，可只安装剩余依赖。

---

## 4. 数据准备

无需手动下载。首次训练会自动下载 CIFAR-10 或 MNIST 到 `data/`。

---

## 5. 训练

```bash
python scripts/train.py --config configs/cifar10_train.yaml
```

快速 smoke（CPU）建议：

```bash
python scripts/train.py --config configs/mnist_quick.yaml
```

输出：
- `outputs/<experiment>/best.pt`
- `outputs/<experiment>/last.pt`
- `outputs/<experiment>/train.log`
- `outputs/<experiment>/history.json`
- `outputs/<experiment>/*.png` 曲线图

---

## 6. 验证与测试

```bash
python scripts/eval.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
python scripts/test.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
```

---

## 7. 导出与量化

### 7.1 TorchScript 导出
```bash
python deployment/export_torchscript.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
```

### 7.2 动态量化（默认启用，针对 Linear 层）
```bash
python deployment/quantize.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
```

> 可选扩展：后续可加 Conv PTQ 或结构化剪枝接口，不影响当前主流程。

### 7.3 一致性检查
```bash
python deployment/consistency_check.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
```

---

## 8. 边缘-云协同演示

### 8.1 分别启动
```bash
python scripts/run_cloud.py --config configs/deploy_cloud.yaml --ckpt outputs/cifar10_full/best.pt
python scripts/run_edge.py --config configs/deploy_edge.yaml --ckpt outputs/cifar10_full/best.pt
```

### 8.2 一键本地双进程演示
```bash
python scripts/run_demo.py --config configs/deploy_edge.yaml --ckpt outputs/cifar10_full/best.pt
```

---

## 9. Benchmark

```bash
python scripts/run_benchmark.py --config configs/cifar10_train.yaml --ckpt outputs/cifar10_full/best.pt
```

输出指标：latency / throughput / params / model_size / cpu_mem。

---

## 10. 论文/答辩可引用系统说明（模板）

本系统构建了“语义编码-信道扰动-语义解码-任务头”的统一研究流水线，在同一代码基中覆盖训练、评估、导出、量化和边缘云协同部署。通过设备画像与预算驱动的分割策略（p0/p1/p2），系统可在不同算力与带宽约束下动态选择推理切分点，并通过一致性检查确保原模型、导出模型与量化模型输出差异可控。

---

## 11. 常见问题排查

1. `ModuleNotFoundError`：请在项目根目录运行命令，或设置 `PYTHONPATH=.`。
2. 下载失败：检查网络或使用镜像源安装依赖。
3. CUDA 不可用：默认可走 CPU 路径；将 config 的 `device` 设为 `cpu`。
4. 端口占用：修改 `configs/deploy_*.yaml` 中的端口。
5. Windows 下多进程 dataloader 问题：可将 `num_workers` 改为 0。

---

## 12. 最快跑通流程（推荐）

1) 训练（快速）
```bash
python scripts/train.py --config configs/mnist_quick.yaml
```
2) 验证
```bash
python scripts/eval.py --config configs/mnist_quick.yaml --ckpt outputs/mnist_quick/best.pt
```
3) 导出
```bash
python deployment/export_torchscript.py --config configs/mnist_quick.yaml --ckpt outputs/mnist_quick/best.pt
```
4) 量化
```bash
python deployment/quantize.py --config configs/mnist_quick.yaml --ckpt outputs/mnist_quick/best.pt
```
5) 一致性
```bash
python deployment/consistency_check.py --config configs/mnist_quick.yaml --ckpt outputs/mnist_quick/best.pt
```
6) benchmark
```bash
python scripts/run_benchmark.py --config configs/mnist_quick.yaml --ckpt outputs/mnist_quick/best.pt
```

# 乳腺肿瘤分类中的概率校准与保形预测

机器学习课程独立数据分析项目。作者：姜广阔，20245621。

本项目使用 scikit-learn 自带的 Wisconsin Diagnostic Breast Cancer 数据，比较逻辑回归、随机森林与直方图梯度提升模型，研究概率校准、类别条件覆盖和测试输入扰动。所有结果由本地实际运行产生。该数据包含从细胞核图像提取的表格特征，项目没有训练医学影像模型，也没有进行临床验证。

## 一键复现

建议 Python 3.12，在独立环境中安装固定版本：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py --repeats 10
python verify_results.py
python make_figures.py
```

`run_experiments.py` 不依赖网络下载数据。随机种子固定为 2026 至 2035，每次划分独立调参，保持四个集合互斥。完整重跑会覆盖 results/ 和 figures/ 的同名运行结果。不同系统、底层数学库和版本可能产生浮点差异。

## 实验设计

- 全部 569 条样本、30 个数值特征；恶性重编码为正类 1，良性为 0。
- 每个种子使用无分层随机排列切分为训练 284、概率校准 85、保形校准 86、测试 114。此处采用无分层排列，使保形校准和测试样本在随机抽样模型下保持可交换。
- LR 在 Pipeline 内标准化；训练集内部五折分层交叉验证，以 ROC-AUC 选择超参数。三个模型使用相同划分。
- 概率校准使用 `sigmoid(a*logit(p)+b)`，约束 a>=0，极小 L2 惩罚 1e-4，p 截断到 [1e-6,1-1e-6] 后进入校准器。此方法不是温度缩放。
- 非一致性分数 `1-p_true`，alpha=0.1；使用 ceil((n+1)*(1-alpha)) 阶次统计量。分别评估全局阈值和按候选标签计算的类别条件阈值。允许空集，空集按未覆盖计，不作强制补全。
- 对测试特征施加 `max(0, X + sigma*train_std*Z)`，sigma 属于 {0,0.25,0.5,1}，同一划分的 Z 在强度间共享。模型和校准器全部冻结。此为人工压力测试，不是真实医院或设备域迁移。
- 输出 ROC-AUC、AP、F1、灵敏度、特异度、Brier、NLL、等宽正类 ECE（5/10/15箱）及预测集覆盖、大小、空集率、单元素错误率。
- 主划分种子预设为 2026；10 次重复划分报告均值和样本标准差。重复划分复用同一数据，不是 10 个外部队列。
- 主划分 LR 与 RF 校准结果进行 2,000 次成对 bootstrap；区间条件于已拟合模型，不涵盖重新训练的不确定性。

## 文件说明

| 文件 | 用途 |
|---|---|
| run_experiments.py | 数据审查、嵌套于训练集的调参、校准、预测与全部数值导出 |
| make_figures.py | 根据保存结果生成报告图表 |
| verify_results.py | 独立重算逐样本结果，核验切分互斥和 250 组指标 |
| data.csv | 导出的全部数据及原始顺序 row_id，无原始患者身份信息 |
| DATA_DESCRIPTION.txt | sklearn 原始数据说明及来源 |
| results/predictions.csv | 27,360 条逐样本预测，含两类保形集合的成员标志 |
| results/splits.csv | 10 次划分的样本编号与类别 |
| results/metrics.csv | 240 行分类和概率评价结果 |
| results/conformal.csv | 480 行预测集评价结果，含阈值及 Wilson 二项近似区间 |
| results/cv_*.csv | 各种子、各模型完整交叉验证搜索记录 |
| results/tuning.json | 每次选择的超参数和训练内部 CV 分数 |
| results/main_predictions.json | 主划分预测、校准参数和保形阈值 |
| results/bootstrap.json | 固定测试集上的成对 bootstrap 比较 |
| results/environment.json | 实际依赖版本、运行耗时、种子与实验设置 |
| results/verification.json | 数值核验结果 |
| figures/ | 实验图表 |
| RUN_LOG.txt | 实际执行日志 |

## 主要结果与边界

主划分校准 LR 的 AUROC 为 0.9940、Brier 为 0.02475；10 次划分 Brier 均值约 0.02433。校准并未普遍改善：RF 的平均 Brier 由约 0.03613 变为 0.03955。LR 全局保形覆盖在主划分仅为 97/114=85.09%，10 次划分均值为 91.93%；边际理论保证不要求每次固定划分都达到 90%。

代码的类别条件 LAC 对每个候选标签分别计算阈值，不等于 Romano 等论文的随机化自适应分数算法。报告提出并验证了严格单调概率变换下类别条件 LAC 集合的不变性；这是对现有方法的应用分析，不声称发明了保形预测。任何空预测集都意味着未覆盖，不能作为模型在该样本上正确的证据。

## 文献与数据

1. Guo C, Pleiss G, Sun Y, Weinberger K Q. On Calibration of Modern Neural Networks. ICML 2017, PMLR 70:1321-1330. https://proceedings.mlr.press/v70/guo17a.html
2. Ovadia Y, et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019, 32. https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html
3. Romano Y, Sesia M, Candes E J. Classification with Valid and Adaptive Coverage. NeurIPS 2020, 33:3581-3591. https://proceedings.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
4. Wolberg W, Mangasarian O, Street N, Street W. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

报告通过比较三篇论文的原始 PDF 撰写，未复现原论文的深度网络大规模基准实验。仓库不包含老师提供的其他同学报告示例或论文 PDF。

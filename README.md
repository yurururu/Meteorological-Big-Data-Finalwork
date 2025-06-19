# Meteorological-Big-Data-Finalwork
Physics-constrained NN (forked form [constrained-downscaling-main](https://github.com/RolnickLab/constrained-downscaling.git))

这个仓库是我为了完成《气象大数据》课程实践而建的

本次实践我按照原论文**Physics-Constrained Deep Learning for Climate Downscaling**开源的代码，在windows上配置环境并进行修改

按照我的报告的逻辑设计了12组实验：
| 编号 | 数据集 | 上采样倍数 | 模型     | 是否处理时序 | 物理约束   | 实验目的简述                             |
|------|--------|-------------|----------|----------------|------------|------------------------------------------|
| E1   | TCW2   | 2×          | CNN      | 否             | None       | 基线模型：简单CNN无约束                 |
| E2   | TCW2   | 2×          | CNN      | 否             | Softmax    | Softmax物理约束对CNN的效果              |
| E3   | TCW4   | 4×          | CNN      | 否             | None       | 基线模型：简单CNN无约束                 |
| E4   | TCW4   | 4×          | CNN      | 否             | Softmax    | Softmax物理约束对CNN的效果              |
| E5   | T1   | 4×          | ConvGRU  | 是             | None       | 基线ConvGRU时序建模性能                 |
| E6   | T1  | 4×          | ConvGRU  | 是             | Softmax    | 加入Softmax约束对ConvGRU的提升          |
| E7   | T1   | 4×          | ConvGRU  | 是             | Soft       | Soft软约束的协同效果                    |
| E8   | TCW4   | 4×          | GAN      | 否             | None       | GAN在无约束条件下性能                   |
| E9   | TCW4   | 4×          | GAN      | 否             | Soft       | GAN结合Soft软约束                       |
| E10  | TCW4   | 4×          | GAN      | 否             | Softmax    | GAN结合Softmax物理一致性约束           |
| E11  | TCW8   | 8×          | GAN      | 否             | None       | 高倍放大下，GAN性能表现                 |
| E12  | TCW8   | 8×          | GAN      | 否             | Softmax    | 高倍放大下，加入Softmax物理约束效果     |

- 由于以上实验得到结果基本可以判定在`TCW4`数据集上`GAN`模型加上`Softmax`约束效果最佳，故报告最后我又加了一组实验，测试`TCW4`数据集在`GAN`模型加上`Add`约束层的效果，效果也是不错的

- 本次试验结果基本可以判定，`CNN、GAN、FlowConvGRU`三种模型再加上物理约束后都有一定的性能提升，但是`CNN`在上采样率为`2x`时，效果不好，可能是由于上采样倍数不够或者数据集数量不够，原因有待验证

- 本次实践中的流程图除了`CNN`的模型架构外均为本人自己制作，实验结果的特定帧的展示也是我自己得出的实验结果截图，本次实践时间跨度不长，但是前前后后也改了好几版，后面模型名称有打错的`tcw`打成了`twc`

- 另外，我作为联合作者也发表了一篇模型中加入物理模块的相关论文`A Physics-Enhanced Network for Predicting Sequential Satellite Images of Typhoon Clouds`，未来会在这个领域继续深耕

# KernelForge 实验记录

范围：formal；协议formal_v1；数值均从原始记录生成。

## 主划分测试集

| 模型 | Accuracy | balanced_accuracy | macroF1 | AUC |
| --- | ---: | ---: | ---: | ---: |
| majority | 0.6316 | 0.5000 | 0.3871 | 0.5000 |
| logistic | 0.9825 | 0.9762 | 0.9810 | 0.9987 |
| own_linear | 0.9737 | 0.9692 | 0.9716 | 0.9980 |
| libsvm_linear | 0.9737 | 0.9692 | 0.9716 | 0.9980 |
| own_rbf | 0.9737 | 0.9792 | 0.9721 | 0.9954 |
| libsvm_rbf | 0.9737 | 0.9792 | 0.9721 | 0.9954 |

## 同精度合成benchmark

| n | 配置 | median ms | min/max ms | baseline/this | 更新数 | 核行计算 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 200 | first_none | 86.156 | 84.071/92.076 | 1.000 | 582 | 1810 |
| 200 | second_none | 84.022 | 83.996/84.157 | 1.025 | 469 | 1585 |
| 200 | second_lru_small | 83.848 | 82.891/84.144 | 1.028 | 469 | 1309 |
| 200 | second_lru | 55.532 | 54.950/55.569 | 1.551 | 469 | 166 |
| 200 | second_full | 53.790 | 51.847/56.161 | 1.602 | 469 | 200 |
| 480 | first_none | 231.141 | 229.385/237.643 | 1.000 | 1136 | 4879 |
| 480 | second_none | 225.048 | 222.578/229.932 | 1.027 | 946 | 4134 |
| 480 | second_lru_small | 227.008 | 226.695/259.848 | 1.018 | 946 | 4106 |
| 480 | second_lru | 121.103 | 119.701/127.831 | 1.909 | 946 | 385 |
| 480 | second_full | 116.119 | 115.920/123.312 | 1.991 | 946 | 480 |
| 800 | first_none | 490.107 | 472.275/506.913 | 1.000 | 1602 | 9019 |
| 800 | second_none | 446.982 | 431.605/473.370 | 1.096 | 1311 | 7236 |
| 800 | second_lru_small | 452.540 | 442.735/460.614 | 1.083 | 1311 | 7226 |
| 800 | second_lru | 207.411 | 205.103/210.679 | 2.363 | 1311 | 619 |
| 800 | second_full | 201.404 | 194.632/218.325 | 2.433 | 1311 | 800 |

## 三划分波动

数值为不同划分的均值±总体标准差，不是同一测试集上的多次训练，也没有统计显著性声明。

- own_linear: balanced_accuracy=0.9620±0.0071; AUC=0.9913±0.0062; splits=3
- own_rbf: balanced_accuracy=0.9487±0.0255; AUC=0.9903±0.0056; splits=3
- logistic: balanced_accuracy=0.9524±0.0194; AUC=0.9913±0.0069; splits=3
- majority: balanced_accuracy=0.5000±0.0000; AUC=0.5000±0.0000; splits=3

## 解释边界

仅一个569×30公共数值特征基准；三次分层划分反映划分波动，不是临床验证或显著性检验。速度仅适用于这里的合成规模、停止容差和本机环境；缓存字节不是进程峰值内存。LIBSVM是同参数独立参考，未声称本实现超过成熟库。

数据：data_source.json；划分：splits.json；选参：validation_candidates.csv、selected_parameters.json；错误分析：test_predictions.csv；停止状态：test_results.json；剖析：profile_summary.json。

## Reproduction boundaries

The current suite has 26 test methods; this snapshot predates adding portable reference verification. The three numerical core files remain byte-identical. The reference manifest records core/result identity. Models are not committed.

`verify --reference` checks saved scores; `experiment` creates a new training/timing run; `verify` checks that run and saved inference.

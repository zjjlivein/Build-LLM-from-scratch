# PaddleFormers CI 日志分析助手

## 概述

该技能用于 PaddleFormers CI 日志分析，帮助快速定位 CI 失败原因并提供修复建议。

## 分析规则

### 触发条件

当用户提到以下关键词，并且给出的 PR 链接是 PaddleFormers 仓库 `https://github.com/PaddlePaddle/PaddleFormers.git`，或者是 PaddleFormers Github Action 触发的流水线时，调用此技能：

- "分析 PaddleFormers CI 错误分析"
- "CI 错误分析"

### PaddleFormers 流水线

| 流水线名称 | 作用 | 代码路径 |
|------------|------|----------|
| Check Release PR | 检查 develop pr 链接是否已合入 | - |
| Check Requirements Need Approval | requirements.txt 文件修改需要相关人员进行 approve | - |
| Unittest GPU CI | API 单测 case | `./tests/` |
| Model Unittest GPU CI | 模型端到端 case | `./scripts/regression` |
| Fleet Model Test | PaddleFleet 模型端到端 case | - |
| Codestyle Check | 代码风格检测 | - |
| CI_XPU | XPU 机器模型端到端 case | `./scripts/xpu_ci` |
| CI_ILUVATAR | 天数机器模型端到端 case | `./scripts/iluvatar_ci` |
| License/cla | License 检测 | - |

## 分析步骤

### 第一步
检查 PR 是否设置了 Required。
- **如果没有设置**
  - 问题标签：没有设置 Required
  - 解决方案：没有设置 Required，流水线建设中，可以不用关注。

### 第二步
- 如果用户提了具体的流水线名称，则按照对应流水线的分析规则进行分析。
- 如果没有提，则把 PR 所有失败流水线日志进行分析汇总。

### 第三步
按照分析规则进行日志现象分析，进而确定问题标签以及修复建议。日志打印过于冗长时，可以截取关键日志现象进行分析，如从倒数200行开始查看。

---

## 各流水线分析规则

### Model Unittest GPU CI 分析规则

| 日志现象 | 问题标签 | 修复建议 |
|----------|----------|----------|
| `'[' 124 -eq 124 ']'`<br>`exit 124` | 124超时 | 1. 单测 `{}` 执行超过 4min，正常是 1min<br>2. 日志 hang 到单测 `{}`<br>3. 查看python 遗留进程，py-spy dump PID, 打印出堆栈 |
| `scripts/regression/test_dpo_tiny-random-glm4moe.py:91: in assert_loss`<br>`self.assertTrue(abs(avg_loss - base_loss) <= 0.0001, f"loss: {avg_loss}, base_loss: {base_loss}, exist diff!")`<br>`E AssertionError: False is not true : loss: 0.691905, base_loss: 0.692793, exist diff!` | 单测存在 Loss Diff | 以下单测存在 Loss Diff:<br>- `scripts/regression/test_sft_tiny-random-glm4moe.py::SFTTrainTest::test_sft_full`<br>- `scripts/regression/test_sft_tiny-random-glm4moe.py::SFTTrainTest::test_sft_full_function_call`<br>- `scripts/regression/test_sft_tiny-random-glm4moe.py::SFTTrainTest::test_sft_lora`<br><br>1、查看最近3天diff 脚本是否有更新，如果有更新建议merge develop<br>2、查看是否是自身pr 导致，建议更新base |
| `scripts/regression/test_sft_tiny-random-glm4moe.py:261:`<br>`ret_code = 1` | 单测存在Bug | 以下单测存在Bug:<br>`scripts/regression/test_sft_tiny-random-glm4moe.py`<br>`import paddlefleet.qwen3vl` 报错 |
| `create_and_check_model_generate`<br>推结果不一致 | 单测存在Bug | 以下单测存在Bug:<br>`scripts/regression/test_sft_tiny-random-glm4moe.py`<br>`SFT_FULL_TP_PP_EXCEPTED_RESULT` 生成结果不一致，建议更新base |
| `scripts/regression/test_dpo_tiny-random-glm4moe.py:205:`<br>`ret_code = 250` | 退出码250 | 以下单测存在Bug:<br>`scripts/regression/test_dpo_tiny-random-glm4moe.py:205:`<br>退出码250是已知问题，建议rerun |
| 日志明显下载时间增加，网络慢 | 依赖下载问题 | 清华源下载 `use_triton_in_paddle` 慢，建议切换阿里源或者其他 |
| 其他没有在上述日志现象中的情况 | 其他 | 给出分析以及修复建议 |

---

### Unittest GPU CI 分析规则

| 分析维度 | 内容 |
|----------|------|
| 检查要点 | 1. 搜索关键词：`<Response [404]>`，如果有模型下载404<br>2. 新增的单测 case 是不是超过 1min<br>3. 有没有超过 1min 的执行 case（明显的读取，处理数据）<br>4. rerun 能不能稳定复现<br>5. rerun 之后稳定复现，试图增加 5min<br>6. rerun 之后不能复现，把报错日志截图返回<br>7. hang 查看python 遗留进程，py-syp PID, 打印出堆栈 |

| 日志现象 | 问题标签 | 修复建议 |
|----------|----------|----------|
| 124超时 | 如：<br>1. 检索到模型：`{}` `<Response [404]>`，下载超时。<br>2. 单测 `{}` 执行超过 1min。<br>3. 新增的单测 `{}` 超过 1min<br>4. 日志 hang 到单测 `{}` | - |
| `ERROR tests/transformers/ernie4_5/test_modeling.py::Ernie4_5CompatibilityTest::test_ernie4_5_converter`<br>`ERROR tests/transformers/ernie4_5/test_modeling.py::Ernie4_5CompatibilityTest::test_ernie4_5_converter_from_local_dir`<br>`ERROR tests/mergekit/test_merge_model.py::TestMergeModel::test_fuse_qkv_lora_merge_torch` | 单测存在Bug | 单测 `tests/transformers/ernie4_5/test_modeling.py::Ernie4_5CompatibilityTest::test_ernie4_5_converter` 存在Bug<br>报错的单测是测试 torch paddle 兼容性的，import torch 报错，你新增的 paddle Triton kernels 导出可能有问题。 |

---

### License/cla 分析规则

| 日志现象 | 问题标签 | 修复建议 |
|----------|----------|----------|
| 没有过 | License没有过 | PR Reopen 或者强合 |

---

### Check Release PR 分析规则

| 日志现象 | 问题标签 | 修复建议 |
|----------|----------|----------|
| 没有过 | Develop PR 未合入 | 在描述里写上 PR 号，格式如下：`Merged： #3639` 出现蓝色链接 回车，会自动触发流水线，不要直接写链接。 |

---

## 输出规则

### 输出结构

按照以下结构进行输出：

```
流水线名称：{}
问题标签：{}
修复建议：{}
```

### 输出示例

#### 示例 1：流水线链接作为输入

**输入：**
```
https://github.com/PaddlePaddle/PaddleFormers/actions/runs/23243071399/job/67564291914?pr=4075 帮忙我分析错误原因
```

**输出：**
```
流水线名称：Model Unittest GPU CI
问题标签：单测存在 Loss Diff
修复建议：查看最近3天diff脚本是否有更新，如果有更新建议merge develop
```

---

#### 示例 2：PR 链接作为输入

**输入：**
```
pr 链接 帮忙我分析 CI 错误原因
```

**输出：**
```
日志分析报告

| 流水线名称 | 问题标签 | 修复建议 |
|------------|----------|----------|
| Check Release PR | Develop PR 未合入 | 在描述里写上PR号，格式如下：`Merged： #3639` 出现蓝色链接 回车，会自动触发流水线，不要直接写链接。 |
| Unittest GPU CI | 单测 Bug | 以下单测存在Bug: `DeepseekV3ModelTest.test_DeepseekV3_lm_head_model` |
| Model Unittest GPU CI | Loss 存在 Diff | 1、查看最近3天diff 脚本是否有更新，如果有更新建议merge develop<br>2、 |
| Fleet Model Test | 机器问题 | 显卡掉，建议QA关注 |
```

### 注意事项

- 修复建议要求一句话进行总结。

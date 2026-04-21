# AI 评测平台使用指南

本文档详细说明 AI 评测平台五大核心模块的使用方法，涵盖 LLM 配置、数据管理、场景指标、评测执行和报告查看的完整流程。

---

## 一、LLM 配置

LLM 配置中心用于集中管理评测所需的大语言模型 API 连接信息。评测引擎通过此处配置的 LLM 来执行基于模型的指标评分。

### 1.1 预配置的通义千问

系统首次启动时自动创建一条通义千问配置：

| 字段 | 值 |
|------|-----|
| 名称 | Qwen Plus (通义千问) |
| 提供商 | openai（兼容协议） |
| API 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 模型名称 | qwen-plus |
| Temperature | 0.01 |
| 最大 Token | 1024 |
| 默认配置 | 是 |

> 使用前需将 `api_key` 更换为您自己的 DashScope API Key。

### 1.2 添加新的 LLM 配置

在「LLM 配置」页面点击「新增 LLM」，填写以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| **名称** | 配置的显示名称，需唯一 | `GPT-4o` |
| **API 端点** | OpenAI 兼容的 API base URL | `https://api.openai.com/v1` |
| **API Key** | 对应端点的认证密钥 | `sk-xxxx` |
| **模型名称** | 要调用的模型 ID | `gpt-4o` |
| **Temperature** | 采样温度，评测建议用低值 | `0.01` |
| **最大 Token** | 单次回复的最大 token 数 | `1024` |
| **是否默认** | 设为默认后创建评测时自动选中 | 是/否 |

#### 支持的 API 端点

平台使用 OpenAI 兼容协议，支持任何兼容的服务商：

- **OpenAI**：`https://api.openai.com/v1`
- **通义千问 (DashScope)**：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- **DeepSeek**：`https://api.deepseek.com/v1`
- **本地模型 (Ollama)**：`http://localhost:11434/v1`
- **OpenRouter**：`https://openrouter.ai/api/v1`

### 1.3 测试连通性

配置保存后，在 LLM 列表中点击「测试连接」按钮。系统会向 API 发送一条简短请求（`Hi`），返回结果包括：

- **成功**：显示 `Connection successful` 以及响应延迟（毫秒）
- **失败**：显示具体错误信息（如 API Key 无效、网络不通等）

> 建议每次修改配置后先测试连通性，再用于评测任务。

---

## 二、数据管理

数据管理中心负责创建数据集、定义字段结构、录入或导入评测数据。

### 2.1 创建数据集

在「数据管理」页面点击「新建数据集」：

1. **填写基本信息**：
   - **名称**：如 `RAG 问答测试集`
   - **描述**：对数据集的简要说明
   - **样本类型** (`sample_type`)：选择 `single_turn`（单轮）或 `multi_turn`（多轮）

2. **定义字段 Schema**：
   根据所选样本类型，系统会预填推荐字段模板。您也可以自定义字段。

#### 单轮场景 (single_turn) 推荐字段

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `user_input` | text | 是 | 用户输入的问题 |
| `response` | text | 是 | 模型生成的回答 |
| `retrieved_contexts` | text_list | 否 | 检索到的上下文列表 |
| `reference` | text | 否 | 参考标准答案 |

#### 多轮场景 (multi_turn) 推荐字段

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `user_input` | conversation | 是 | 多轮对话内容（JSON 格式） |
| `reference` | text | 否 | 参考标准答案 |
| `reference_tool_calls` | tool_call_list | 否 | 期望的工具调用列表 |

#### 支持的字段类型

| 类型 | 说明 | 数据格式 |
|------|------|----------|
| `text` | 普通文本 | 字符串 |
| `number` | 数值 | 整数或浮点数 |
| `text_list` | 文本列表 | JSON 字符串数组，如 `["ctx1", "ctx2"]` |
| `conversation` | 多轮对话 | JSON 消息数组（见下文） |
| `tool_call_list` | 工具调用列表 | JSON 对象数组（见下文） |
| `tags` | 标签 | JSON 字符串数组 |
| `json` | 自由 JSON | 任意 JSON 结构 |

### 2.2 conversation 字段数据格式

多轮对话使用消息数组表示，每条消息包含 `type` 和 `content` 字段：

```json
[
  {
    "type": "HumanMessage",
    "content": "帮我查一下北京明天的天气"
  },
  {
    "type": "AIMessage",
    "content": "好的，我来帮您查询北京明天的天气。",
    "tool_calls": [
      {
        "name": "get_weather",
        "args": {"city": "北京", "date": "tomorrow"}
      }
    ]
  },
  {
    "type": "ToolMessage",
    "content": "北京明天晴，气温 15-25 度"
  },
  {
    "type": "AIMessage",
    "content": "根据查询结果，北京明天天气晴朗，气温在 15 到 25 度之间。建议穿轻薄外套出行。"
  }
]
```

消息类型说明：
- **HumanMessage**：用户发送的消息
- **AIMessage**：AI 助手的回复，可包含 `tool_calls` 字段表示调用工具
- **ToolMessage**：工具执行返回的结果

### 2.3 tool_call_list 字段数据格式

工具调用列表用于记录期望的工具调用序列：

```json
[
  {
    "name": "get_weather",
    "args": {"city": "北京", "date": "tomorrow"}
  },
  {
    "name": "send_notification",
    "args": {"message": "天气已查询完成"}
  }
]
```

### 2.4 导入 CSV/JSON 数据

在数据集详情页点击「导入数据」，支持两种格式：

#### CSV 格式

```csv
user_input,response,reference
Python中列表和元组有什么区别？,列表可变元组不可变,列表是可变的有序集合元组是不可变的有序集合
什么是Python的装饰器？,装饰器是特殊函数可以增强其他函数,装饰器是一种设计模式允许扩展函数的行为
```

> CSV 中的 `text_list` 类型字段，值应为 JSON 字符串，例如：`"[""ctx1"", ""ctx2""]"`

#### JSON 格式

```json
[
  {
    "user_input": "Python中列表和元组有什么区别？",
    "response": "列表可变，元组不可变。",
    "retrieved_contexts": ["Python列表(list)是可变序列", "Python元组(tuple)是不可变序列"],
    "reference": "列表是可变的有序集合，元组是不可变的有序集合。"
  }
]
```

> JSON 文件必须是对象数组格式（最外层为 `[...]`）。

### 2.5 查看和编辑数据

- 在数据集详情页可以浏览数据行，支持分页查看（默认每页 20 条，最多 100 条）
- 点击行可查看完整数据内容
- 支持通过「添加行」按钮逐行录入新数据
- 支持删除单行数据（删除后自动更新行计数 `row_count`）

---

## 三、场景与指标管理

场景定义了一次评测中使用哪些指标、每个指标的权重和通过阈值。

### 3.1 三个预置模板

#### RAG 评测模板

适用于检索增强生成（RAG）系统的评测。

| 指标 | 说明 | 通过阈值 | 权重 |
|------|------|----------|------|
| Faithfulness | 回答是否基于检索上下文，不捏造信息 | 0.7 | 1.0 |
| Context Recall | 参考答案的要点是否在检索上下文中有覆盖 | 0.7 | 1.0 |
| Factual Correctness | 回答与参考答案在事实层面的一致性 | 0.7 | 1.0 |

**需要的数据字段**：`user_input`、`response`、`retrieved_contexts`、`reference`

#### Agent 评测模板

适用于 AI Agent 智能体系统的评测。

| 指标 | 说明 | 通过阈值 | 权重 |
|------|------|----------|------|
| Tool Call Accuracy | Agent 是否正确调用了工具（名称和参数） | 0.8 | 1.0 |
| Agent Goal Accuracy | Agent 是否完成了用户的目标 | 0.8 | 1.0 |

**需要的数据字段**：`user_input`（conversation 格式）、`reference_tool_calls`

#### 多轮对话评测模板

适用于多轮对话系统的评测。

| 指标 | 说明 | 通过阈值 | 权重 |
|------|------|----------|------|
| Topic Adherence | 对话是否始终围绕主题 | 0.7 | 1.0 |
| Coherence | 回答是否逻辑连贯、条理清晰 | 0.5 | 1.0 |

**需要的数据字段**：`user_input`（conversation 格式）

### 3.2 创建自定义场景

在「场景管理」页面点击「新建场景」：

1. 填写场景名称、描述、场景类型（`rag` / `agent` / `multi_turn`）
2. 选择数据类型（`single_turn` 或 `multi_turn`）
3. 从指标库中选择指标，为每个指标设置：
   - **权重**：指标在综合评分中的权重（默认 1.0）
   - **通过阈值**：分数达到此值即视为该指标通过（如 0.7 表示 70 分及格）

### 3.3 内置指标库

平台预置了 10 个评测指标，按类别分组：

#### RAG 类

| 指标名 | 技术类型 | 说明 |
|--------|----------|------|
| Faithfulness | builtin_faithfulness | 回答忠实度，是否基于上下文、无幻觉 |
| Context Recall | builtin_context_recall | 上下文召回率，参考答案要点是否被检索覆盖 |
| Context Precision | builtin_context_precision | 上下文精确度，检索内容中有用信息的占比 |
| Answer Relevancy | builtin_answer_relevancy | 回答相关性，回答与问题的匹配程度 |
| Factual Correctness | builtin_factual_correctness | 事实正确性，回答与参考答案的事实一致性 |

#### Agent 类

| 指标名 | 技术类型 | 说明 |
|--------|----------|------|
| Tool Call Accuracy | builtin_tool_call_accuracy | 工具调用准确率，名称和参数是否正确 |
| Agent Goal Accuracy | builtin_agent_goal_accuracy | 目标完成准确率，是否达成用户意图 |

#### 多轮对话类

| 指标名 | 技术类型 | 说明 |
|--------|----------|------|
| Topic Adherence | builtin_topic_adherence | 主题贴合度，对话是否围绕主题 |

#### 通用自定义类

| 指标名 | 技术类型 | 说明 |
|--------|----------|------|
| Harmfulness | aspect_critic | 回答是否有害（基于 LLM 判断） |
| Coherence | aspect_critic | 回答是否逻辑连贯 |

### 3.4 创建自定义指标

在「指标管理」中点击「新建指标」，支持以下类型：

#### AspectCritic（二值判断型）

让 LLM 根据自定义标准判断是否满足某个方面的要求，返回 0 或 1。

```json
{
  "name": "politeness",
  "display_name": "礼貌性",
  "metric_type": "aspect_critic",
  "config": {
    "definition": "Does the response maintain a polite and respectful tone throughout? Consider greeting words, sentence structure, and overall attitude."
  },
  "category": "custom"
}
```

`config.definition` 是评判标准的英文描述，LLM 会根据此标准对回答进行二值判断。

#### DiscreteMetric（离散值型）

让 LLM 从预定义的选项中选择一个评分。

```json
{
  "name": "quality_level",
  "display_name": "质量等级",
  "metric_type": "discrete_metric",
  "config": {
    "values": ["excellent", "good", "fair", "poor"],
    "definition": "Rate the overall quality of the response."
  },
  "category": "custom"
}
```

#### NumericMetric（数值型）

让 LLM 在指定范围内给出数值评分。

```json
{
  "name": "detail_score",
  "display_name": "详细程度",
  "metric_type": "numeric_metric",
  "config": {
    "min_value": 0,
    "max_value": 10,
    "definition": "Rate how detailed and comprehensive the response is on a scale of 0 to 10."
  },
  "category": "custom"
}
```

### 3.5 指标分类说明

| 分类 | 说明 |
|------|------|
| **内置指标 (is_builtin=true)** | 由 Ragas 框架提供的标准指标，不可删除 |
| **自定义指标 (is_builtin=false)** | 用户创建的指标，可删除 |
| **LLM-based** | 需要调用 LLM 进行评分（大部分指标属于此类） |
| **Non-LLM** | 不需要 LLM 即可计算的指标（如字符串相似度） |

---

## 四、评测执行

### 4.1 创建评测任务

在「评测执行」页面：

1. 输入**任务名称**：如 `RAG 系统 v2.0 评测`
2. **选择数据集**：从已创建的数据集中选择
3. **选择场景**：从预置或自定义场景中选择
4. **选择 LLM 配置**：选择用于评分的 LLM
5. 点击「开始评测」

> 确保数据集的字段与场景所需指标的输入字段匹配。例如 RAG 场景需要 `user_input`、`response`、`retrieved_contexts`、`reference` 字段。

### 4.2 离线评测模式

当前版本采用离线评测模式（Offline Evaluation）：

- 数据集中的 `response` 字段是**预先生成好的**模型回答
- 评测引擎**不会**实时调用被测系统，而是对已有的回答进行质量评分
- LLM 配置中选择的模型用作「评判模型」（Judge LLM），负责对回答打分

这意味着您需要：
1. 先用被测系统生成回答，将结果填入数据集的 `response` 字段
2. 然后使用评测平台对这些回答进行质量评估

### 4.3 执行过程

评测任务创建后自动开始异步执行：

1. **初始化**：加载数据集、场景配置、LLM 配置，构建 Ragas LLM 实例
2. **逐行评测**：对数据集中每一行数据：
   - 将数据映射为 Ragas Sample（SingleTurnSample 或 MultiTurnSample）
   - 对场景中的每个指标逐一计算分数
   - 根据通过阈值判断每个指标是否通过
   - 所有指标都通过则该行通过（`is_pass=True`）
   - 记录每个指标的分数（`score`）和评分理由（`reason`）
   - 记录评测耗时（`execution_time_ms`）
3. **汇总统计**：计算每个指标的均值（`mean`）、最小值（`min`）、最大值（`max`）、通过率（`pass_rate`）
4. **完成**：更新任务状态为 `completed`，写入 `summary_scores`

### 4.4 监控进度

评测任务列表页面显示每个任务的：

| 字段 | 说明 |
|------|------|
| 状态 (`status`) | `pending`（等待中）-> `running`（执行中）-> `completed`（已完成）/ `failed`（失败）/ `cancelled`（已取消） |
| 进度 (`progress`) | 0.0 ~ 1.0，表示已完成行数的比例 |
| 已完成行数 / 总行数 | 如 `3/5` |
| 错误信息 | 如果失败，显示错误原因 |

### 4.5 取消评测

对于正在执行的任务，可以点击「取消」按钮。取消后：
- 任务状态变为 `cancelled`
- 已评测完的行结果会保留
- 未评测的行不会继续执行

---

## 五、评测报告

### 5.1 总览面板

评测完成后进入报告详情页，总览面板展示：

| 项目 | 说明 |
|------|------|
| **基本信息** | 任务名称、数据集、场景、LLM 配置、开始/结束时间 |
| **通过率** (`pass_rate`) | 通过行数 / 总行数（如 0.8 表示 80%） |
| **通过/失败/错误数** | `pass_count` / `fail_count` / `error_count` |
| **各指标得分** (`metric_summary`) | 每个指标的 `mean`、`min`、`max`、`pass_rate` |

#### 示例报告总览

```
任务：RAG 示例评测
状态：已完成
总行数：5
通过：4 行 | 失败：1 行 | 错误：0 行
通过率：80.00%

指标明细：
+-------------------------+------+------+------+----------+
| 指标                    | 均值 | 最小 | 最大 | 通过率   |
+-------------------------+------+------+------+----------+
| Faithfulness            | 0.87 | 0.67 | 1.00 | 80%      |
| Context Recall          | 0.93 | 0.80 | 1.00 | 100%     |
| Factual Correctness     | 0.80 | 0.50 | 1.00 | 80%      |
+-------------------------+------+------+------+----------+
```

### 5.2 逐条明细

报告页的「明细」标签展示每条数据的评测结果：

| 列 | 说明 |
|----|------|
| 行号 (`row_index`) | 数据在数据集中的索引 |
| 通过/失败 (`is_pass`) | 该行是否所有指标都达到阈值 |
| 各指标得分 (`metric_scores`) | 每个指标的 `score` 和 `reason` |
| 耗时 (`execution_time_ms`) | 该行评测耗费的时间（毫秒） |
| 原始数据 (`dataset_row`) | 关联的数据集行内容 |

#### 单行结果示例

```
行 #3 -- 失败

指标得分：
  Faithfulness:        0.67  <-- 未达阈值 0.7
    理由：回答提到"多线程无法真正利用多核CPU"在上下文中仅有间接支持，有一定的延伸推理

  Context Recall:      0.80  [通过]
    理由：参考答案中关于"限制并行性能"的表述在上下文中有部分体现

  Factual Correctness: 0.50  <-- 未达阈值 0.7
    理由：回答基本正确但过于绝对化，未提及多线程在IO密集型任务中仍然有效

原始数据：
  user_input: Python的GIL是什么？
  response:   GIL是全局解释器锁，它确保同一时间只有一个线程执行Python字节码...
  reference:  GIL（全局解释器锁）是CPython的一个机制...
```

### 5.3 按通过/失败筛选

报告明细支持按状态筛选行：

- **全部**：显示所有行（默认）
- **通过 (`status=pass`)**：只显示所有指标都通过的行
- **失败 (`status=fail`)**：只显示至少一个指标未通过的行
- **错误 (`status=error`)**：只显示评测过程中出错的行

通过 API 查询参数或页面上的筛选按钮进行切换。支持分页，默认每页 20 条。

### 5.4 如何解读指标分数

| 分数范围 | 含义 |
|----------|------|
| **0.9 ~ 1.0** | 优秀 -- 质量很高，几乎没有问题 |
| **0.7 ~ 0.9** | 良好 -- 基本达标，少量瑕疵 |
| **0.5 ~ 0.7** | 一般 -- 有明显改进空间 |
| **< 0.5** | 较差 -- 需要重点关注和改进 |

> 实际判定标准以场景中设置的 `pass_threshold` 为准。不同指标可以设置不同的阈值。

**指标得分中的 `reason` 字段**尤为重要，它是 LLM 给出的评分依据，可以帮助您理解为什么某行数据得到了较低的分数，从而有针对性地改进被测系统。

---

## 六、场景示例

### 6.1 RAG 评测示例

完整的端到端 RAG 评测流程。

#### 第一步：配置 LLM

使用系统预置的通义千问配置（或添加您自己的 LLM 配置），确保 API Key 有效。可点击「测试连接」确认连通。

#### 第二步：准备数据集

1. 创建一个 `single_turn` 类型的数据集，命名为「RAG 问答测试集」
2. 定义字段：
   - `user_input` (text, 必填) -- 用户输入的问题
   - `response` (text, 必填) -- 模型生成的回答
   - `retrieved_contexts` (text_list) -- 检索到的上下文列表
   - `reference` (text) -- 参考标准答案

3. 导入以下 CSV 数据（或手动添加）：

```csv
user_input,response,retrieved_contexts,reference
什么是机器学习？,机器学习是人工智能的一个分支让计算机通过数据学习模式,"[""机器学习是AI的子领域通过算法从数据中学习"", ""监督学习非监督学习和强化学习是三种主要范式""]",机器学习是人工智能的分支使计算机能够从数据中自动学习和改进
深度学习和机器学习有什么关系？,深度学习是机器学习的一个子集使用深层神经网络,"[""深度学习基于多层人工神经网络"", ""深度学习是机器学习的特殊形式""]",深度学习是机器学习的子集通过多层神经网络实现特征自动提取
```

#### 第三步：选择场景

选择预置的「RAG 评测模板」，包含三个指标：
- Faithfulness（通过阈值 0.7）
- Context Recall（通过阈值 0.7）
- Factual Correctness（通过阈值 0.7）

#### 第四步：执行评测

点击「开始评测」，系统自动异步执行。可在评测列表查看进度。

#### 第五步：查看报告

进入报告页面，查看：
- 总览面板：通过率、各指标平均分
- 逐条明细：每行数据的得分和理由
- 重点关注失败行的 `reason` 字段，分析改进方向

### 6.2 Agent 评测示例

Agent 评测需要使用多轮对话格式的数据，评估 Agent 的工具调用能力和目标完成情况。

#### 数据准备

创建 `multi_turn` 类型的数据集，字段包括：

- `user_input` (conversation, 必填)：多轮对话记录
- `reference` (text)：参考标准答案
- `reference_tool_calls` (tool_call_list)：期望的工具调用

示例数据行（通过「添加行」按钮手动录入 JSON）：

```json
{
  "user_input": [
    {
      "type": "HumanMessage",
      "content": "帮我查一下上海今天的天气，然后发送给张三"
    },
    {
      "type": "AIMessage",
      "content": "好的，我先查询上海的天气。",
      "tool_calls": [
        {"name": "get_weather", "args": {"city": "上海"}}
      ]
    },
    {
      "type": "ToolMessage",
      "content": "上海今天多云，25度"
    },
    {
      "type": "AIMessage",
      "content": "上海今天多云 25 度，我现在把这个信息发送给张三。",
      "tool_calls": [
        {"name": "send_message", "args": {"to": "张三", "content": "上海今天多云，25度"}}
      ]
    },
    {
      "type": "ToolMessage",
      "content": "消息发送成功"
    },
    {
      "type": "AIMessage",
      "content": "已经查到上海今天多云 25 度，并成功发送给张三了。"
    }
  ],
  "reference": "成功查询天气并发送给指定联系人",
  "reference_tool_calls": [
    {"name": "get_weather", "args": {"city": "上海"}},
    {"name": "send_message", "args": {"to": "张三", "content": "上海今天多云，25度"}}
  ]
}
```

#### 选择场景

使用预置的「Agent 评测模板」，包含：
- Tool Call Accuracy（通过阈值 0.8）-- 评估工具调用是否正确
- Agent Goal Accuracy（通过阈值 0.8）-- 评估是否完成用户目标

### 6.3 自定义指标示例：礼貌性评测

假设您需要评测客服系统的回答是否够礼貌。

#### 第一步：创建自定义指标

在「指标管理」页面点击「新建指标」，填写：

- **名称**：`politeness`
- **显示名称**：`礼貌性`
- **指标类型**：`aspect_critic`
- **配置**：
  ```json
  {
    "definition": "Does the response maintain a polite, warm, and respectful tone? Consider: (1) Use of greeting/closing phrases (2) Respectful language (3) Empathetic and helpful attitude (4) Avoiding blunt or dismissive language."
  }
  ```
- **分类**：`custom`

或通过 API：

```bash
curl -X POST http://localhost:8000/api/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "name": "politeness",
    "display_name": "礼貌性",
    "metric_type": "aspect_critic",
    "config": {
      "definition": "Does the response maintain a polite, warm, and respectful tone? Consider: (1) Use of greeting/closing phrases (2) Respectful language (3) Empathetic and helpful attitude (4) Avoiding blunt or dismissive language."
    },
    "category": "custom"
  }'
```

#### 第二步：创建自定义场景

创建一个使用此指标的场景：

```bash
curl -X POST http://localhost:8000/api/scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "name": "客服礼貌性评测",
    "description": "评测客服回答的礼貌程度",
    "scene_type": "rag",
    "sample_type": "single_turn",
    "metrics": [
      {
        "metric_definition_id": 11,
        "weight": 1.0,
        "pass_threshold": 0.5
      }
    ]
  }'
```

> AspectCritic 返回 0（不满足）或 1（满足），阈值设为 0.5 意味着评判为 1 时才通过。

#### 第三步：准备数据并执行

数据集只需要 `user_input` 和 `response` 两个字段：

```csv
user_input,response
请问我的快递到了吗？,你好！我帮您查一下哈。经查询您的快递目前已在派送中预计今天下午到达。还有其他需要帮助的吗？
我要退货,把订单号发过来。
这个产品怎么用？,非常感谢您的咨询！这款产品的使用方法如下...如果您在使用过程中有任何问题随时联系我们。
```

执行评测后，第一条和第三条数据可能得到 1 分（礼貌），第二条可能得到 0 分（语气生硬），从而帮助定位客服回答质量问题。

---

## 附录：API 快速参考

| 模块 | 方法 | 路径 | 说明 |
|------|------|------|------|
| LLM 配置 | GET | `/api/llm-configs` | 列表 |
| | POST | `/api/llm-configs` | 创建 |
| | GET | `/api/llm-configs/{id}` | 详情 |
| | PUT | `/api/llm-configs/{id}` | 更新 |
| | DELETE | `/api/llm-configs/{id}` | 删除 |
| | POST | `/api/llm-configs/{id}/test` | 测试连接 |
| 数据集 | GET | `/api/datasets` | 列表 |
| | POST | `/api/datasets` | 创建 |
| | GET | `/api/datasets/{id}` | 详情 |
| | PUT | `/api/datasets/{id}` | 更新 |
| | DELETE | `/api/datasets/{id}` | 删除 |
| | GET | `/api/datasets/{id}/rows?page=1&page_size=20` | 分页查看行 |
| | POST | `/api/datasets/{id}/rows` | 添加行 |
| | DELETE | `/api/datasets/{id}/rows/{row_id}` | 删除行 |
| | POST | `/api/datasets/{id}/import` | 导入 CSV/JSON |
| 指标 | GET | `/api/metrics` | 列表 |
| | POST | `/api/metrics` | 创建 |
| | DELETE | `/api/metrics/{id}` | 删除（仅自定义指标） |
| 场景 | GET | `/api/scenarios` | 全部列表 |
| | GET | `/api/scenarios/presets` | 预置列表 |
| | POST | `/api/scenarios` | 创建 |
| | GET | `/api/scenarios/{id}` | 详情 |
| | DELETE | `/api/scenarios/{id}` | 删除（仅非预置） |
| 评测 | GET | `/api/evaluations` | 列表 |
| | POST | `/api/evaluations` | 创建并自动执行 |
| | GET | `/api/evaluations/{id}` | 详情 |
| | POST | `/api/evaluations/{id}/cancel` | 取消 |
| 报告 | GET | `/api/reports/{eval_id}/summary` | 总览 |
| | GET | `/api/reports/{eval_id}/rows?page=1&page_size=20&status=pass` | 分页明细（可筛选） |
| | GET | `/api/reports/{eval_id}/rows/{row_id}` | 行详情 |

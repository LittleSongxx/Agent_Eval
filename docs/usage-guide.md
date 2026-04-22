# AI 评测平台操作手册

> 本手册以实际操作为导向，按照"启动 → 验证 → 评测"的顺序编写，可直接对照操作。

---

## 零、启动与验证

### 0.1 启动后端

```bash
cd ai-eval-platform/backend
source venv/bin/activate
python3 run.py
```

**验证后端**：打开 http://localhost:8000/docs ，看到 Swagger API 文档页面。

### 0.2 启动前端

```bash
cd ai-eval-platform/frontend
npm run dev
```

**验证前端**：打开 http://localhost:5175 ，看到左侧有 5 个导航菜单的评测平台界面。

### 0.3 验证预置数据

按顺序点击以下页面确认预置数据正常加载：

| 页面 | 应该看到 |
|------|---------|
| LLM 配置 | 一条 "Qwen Plus (通义千问)" 配置 |
| 数据管理 | 3 个示例数据集（见下表） |
| 场景管理 | 3 个预置模板卡片（RAG / Agent / 多轮对话） |
| 评测报告 | 一条已完成的 "RAG 示例评测" |

**3 个示例数据集**：

| 数据集 | 类型 | 条数 | 覆盖字段 | 对应场景 |
|--------|------|------|---------|---------|
| RAG 示例数据集 | single_turn | 5 | user_input, response, retrieved_contexts(text_list), reference | RAG 评测模板 |
| Agent 工具调用示例数据集 | multi_turn | 3 | user_input(conversation), reference, reference_tool_calls(tool_call_list) | Agent 评测模板 |
| 多轮对话示例数据集 | multi_turn | 3 | user_input(conversation), reference, reference_topics(text_list) | 多轮对话评测模板 |

点进每个数据集的详情页查看数据内容，可以直观理解每种字段类型的实际数据格式。

### 0.4 测试 LLM 连通性

1. 进入「LLM 配置」页面
2. 找到 "Qwen Plus (通义千问)" 这行
3. 点击「测试连接」按钮
4. 等待 2-5 秒，应该看到绿色提示 "连接测试成功 (xxxms)"

**如果失败**：编辑 `backend/.env`，检查 `LLM_API_KEY` 是否正确，重启后端。

---

## 一、使用预置数据快速体验评测

最快了解平台的方式：用预置的示例数据跑一次完整评测。

### 步骤 1：创建评测任务

1. 点击左侧「评测执行」
2. 在顶部表单中填写：
   - **任务名称**：`我的第一次评测`
   - **选择数据集**：下拉选 `RAG 示例数据集`
   - **选择场景**：下拉选 `RAG 评测模板`
   - **选择 LLM**：下拉选 `Qwen Plus (通义千问)`
3. 点击「开始评测」

### 步骤 2：等待执行

- 页面下方的任务列表会出现你的任务
- 状态从 `pending` → `running`（显示进度条）→ `completed`
- 5 条数据大约需要 30 秒 ~ 2 分钟（取决于 LLM 响应速度）
- 页面每 3 秒自动刷新进度

### 步骤 3：查看报告

评测完成后：
1. 在任务列表中点击「查看报告」
2. 或点击左侧「评测报告」，选择刚完成的评测

**报告总览 Tab**：
- 4 个统计卡片：通过率、总条数、通过数、不通过数
- 各指标得分表：Faithfulness / Context Recall / Context Precision / Answer Relevancy / Factual Correctness / Answer Completeness 的均值、最小、最大、通过率
- 基本信息：任务名称、数据集、场景、LLM

**逐条明细 Tab**：
- 每行数据的各指标得分和通过/失败状态
- 点击行展开抽屉，查看：
  - 原始数据（user_input、response、reference 等）
  - 每个指标的具体分数和评分理由

---

## 二、创建自己的 RAG 评测

### 2.1 准备数据文件

创建一个 CSV 文件 `my_rag_test.csv`：

```csv
user_input,response,retrieved_contexts,reference,retrieved_context_ids,reference_context_ids
什么是机器学习？,机器学习是让计算机从数据中自动学习的技术,"[""机器学习是AI的子领域"", ""通过算法从数据中学习模式""]",机器学习是人工智能的分支使计算机能从数据中学习,"[""ml-doc-001"", ""ml-doc-002""]","[""ml-doc-001""]"
什么是深度学习？,深度学习使用多层神经网络处理复杂数据,"[""深度学习基于多层神经网络"", ""属于机器学习的子集""]",深度学习是机器学习的子集通过多层神经网络提取特征,"[""dl-doc-001"", ""ml-doc-003""]","[""dl-doc-001""]"
```

**字段说明**：
- `user_input`：用户问题
- `response`：**你的系统生成的回答**（评测对象）
- `retrieved_contexts`：检索到的上下文（JSON 数组格式）
- `reference`：标准参考答案
- `retrieved_context_ids`：检索到的文档/分片 ID，顺序与 `retrieved_contexts` 一致，用于 HitRate@K、MRR 等检索单测指标
- `reference_context_ids`：人工标注的标准文档/分片 ID，用于判断检索是否命中正确资料

> CSV 中的 `retrieved_contexts`、`retrieved_context_ids`、`reference_context_ids` 字段需要用 JSON 数组格式：`"[""文本1"", ""文本2""]"`。
> 如果暂时没有文档 ID，也可以先不填 ID 字段；默认 RAG 问答模板不会强制启用 ID 指标。

### 2.2 创建数据集并导入

1. 点击「数据管理」→「新建数据集」
2. 填写：
   - 名称：`我的 RAG 测试集`
   - 样本类型：选 `single_turn`
   - 字段自动预填（user_input、response、retrieved_contexts、reference），直接确认
3. 点击进入数据集详情页
4. 点击「导入数据」→ 上传 `my_rag_test.csv` → 确认
5. 确认导入后看到数据行出现在表格中

### 2.3 执行评测

1. 点击「评测执行」
2. 选择数据集：`我的 RAG 测试集`
3. 选择场景：`RAG 评测模板`
4. 选择 LLM：`Qwen Plus (通义千问)`
5. 点击「开始评测」

### 2.4 查看结果

评测完成后查看报告，重点关注：
- **Faithfulness 低分的行**：说明回答中有不在检索上下文中的信息（幻觉）
- **Context Recall 低分的行**：说明检索的内容没有覆盖参考答案的要点
- **Context Precision 低分的行**：说明检索结果混入了较多无关内容
- **Answer Relevancy 低分的行**：说明回答没有紧扣用户问题
- **Answer Completeness 低分的行**：说明回答事实可能没错，但遗漏了参考答案中的关键要点
- **reason 字段**：LLM 给出的评分依据，帮你定位具体问题

---

## 三、创建 Agent 评测

Agent 评测评估工具调用准确性和目标达成度。它和“多轮对话评测”的区别是：Agent 场景关注行动链路是否正确，包括是否调用了正确工具、参数是否正确、工具返回后是否完成用户目标；多轮对话场景关注对话体验是否稳定，包括是否跑题、是否记住上下文、回答是否连贯。

| 场景 | 关注点 | 典型字段 | 典型指标 |
|------|--------|----------|----------|
| Agent 评测 | 工具调用和任务闭环 | `reference_tool_calls`, `reference` | Tool Call Accuracy, Agent Goal Accuracy |
| 多轮对话评测 | 话题范围和上下文连贯 | `reference_topics`, 可选 `reference` | Topic Adherence, Coherence |

如果一个样本既是多轮，又包含工具调用和明确任务目标，优先按 Agent 评测处理；如果只是普通 human/ai 多轮聊天，没有工具执行链，就按多轮对话评测处理。

### 3.1 准备数据文件

创建 `my_agent_test.json`：

```json
[
  {
    "user_input": [
      {"type": "human", "content": "帮我查一下订单 ORD-001 的状态"},
      {"type": "ai", "content": "", "tool_calls": [{"name": "query_order", "args": {"order_id": "ORD-001"}}]},
      {"type": "tool", "content": "{\"status\": \"shipped\", \"tracking\": \"SF001\"}"},
      {"type": "ai", "content": "您的订单 ORD-001 已发货，快递单号 SF001。"}
    ],
    "reference": "成功查询到订单状态并告知用户",
    "reference_tool_calls": [
      {"name": "query_order", "args": {"order_id": "ORD-001"}}
    ]
  },
  {
    "user_input": [
      {"type": "human", "content": "帮我取消订单 ORD-002"},
      {"type": "ai", "content": "好的，我帮您取消订单。", "tool_calls": [{"name": "cancel_order", "args": {"order_id": "ORD-002"}}]},
      {"type": "tool", "content": "{\"success\": true}"},
      {"type": "ai", "content": "订单 ORD-002 已成功取消。"}
    ],
    "reference": "成功取消指定订单",
    "reference_tool_calls": [
      {"name": "cancel_order", "args": {"order_id": "ORD-002"}}
    ]
  }
]
```

**关键格式**：
- `user_input`：消息数组，type 为 `human` / `ai` / `tool`
- `ai` 消息的 `tool_calls`：Agent 实际调用的工具
- `reference_tool_calls`：期望 Agent 应该调用的工具

### 3.2 创建数据集

1. 「数据管理」→「新建数据集」
2. 名称：`Agent 工具调用测试集`
3. 样本类型：`multi_turn`
4. 确认预填字段（user_input、reference、reference_tool_calls）
5. 进入详情页 → 「导入数据」→ 上传 JSON 文件

### 3.3 执行评测

- 场景选 `Agent 评测模板`（含 Tool Call Accuracy + Agent Goal Accuracy）
- 执行后查看报告
- **Tool Call Accuracy**：后端确定性比较实际工具调用和期望工具调用，给出匹配数量说明
- **Agent Goal Accuracy**：平台原生 Judge 阅读完整 Agent 轨迹，判断是否达成用户目标并返回理由

### 3.4 创建多轮对话评测

多轮对话评测不要求工具调用，重点看对话是否围绕业务话题、上下文是否连贯。

核心字段：

- `user_input`：human/ai 交替的完整对话
- `reference_topics`：允许讨论的话题范围，例如 `["订单查询", "退货退款", "物流追踪"]`
- `reference`：可选，用于描述期望对话结果

执行时选择 `多轮对话评测模板`：

- **Topic Adherence**：平台原生 Judge 判断对话是否围绕 `reference_topics`
- **Coherence**：平台原生 Judge 判断对话是否前后连贯、逻辑清晰

---

## 四、创建自定义指标

### 4.1 通过 API 创建（推荐）

创建一个"礼貌性"评测指标：

```bash
curl -X POST http://localhost:8000/api/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "name": "politeness",
    "display_name": "礼貌性",
    "metric_type": "aspect_critic",
    "config": {
      "definition": "Does the response maintain a polite and respectful tone? Return 1 if polite, 0 if not."
    },
    "category": "custom"
  }'
```

创建后在「场景管理」的「新建场景」中即可选择这个指标。

### 4.2 指标类型说明

| 类型 | metric_type | 说明 | 输出 |
|------|------------|------|------|
| 维度评判 | `aspect_critic` | LLM 按自定义标准做二值判断 | 0 或 1 |
| 离散评分 | `discrete` | LLM 从预定义选项中选择 | "pass"/"fail" 等 |
| 数值评分 | `numeric` | LLM 给出数值评分 | 0.0 ~ 1.0 |

### 4.3 内置指标一览

| 指标 | 适用场景 | 需要的数据字段 |
|------|---------|---------------|
| Faithfulness | RAG | user_input, response, retrieved_contexts |
| Context Recall | RAG | user_input, retrieved_contexts, reference |
| Context Precision | RAG | user_input, retrieved_contexts, reference |
| Factual Correctness | RAG | response, reference |
| Answer Relevancy | RAG | user_input, response |
| Answer Completeness | RAG | user_input, response, reference |
| HitRate@K | RAG 检索单测 | retrieved_context_ids, reference_context_ids |
| MRR | RAG 检索单测 | retrieved_context_ids, reference_context_ids |
| Tool Call Accuracy | Agent | user_input (conversation), reference_tool_calls |
| Agent Goal Accuracy | Agent | user_input (conversation), reference |
| Topic Adherence | 多轮对话 | user_input (conversation), reference_topics |
| Harmfulness | 通用 | response |
| Coherence | 通用 | response |

---

## 五、LLM 配置管理

### 5.1 添加新的 LLM

1. 「LLM 配置」→「新增 LLM」
2. 填写字段：

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 名称 | `GPT-4o` | 显示名称（需唯一） |
| API 地址 | `https://api.openai.com/v1` | OpenAI 兼容端点 |
| API Key | `sk-xxx` | 认证密钥 |
| 模型名称 | `gpt-4o` | 模型 ID |
| 温度 | `0.01` | 评测建议低值 |
| 最大 Token | `1024` | 单次回复上限 |

3. 保存后点击「测试连接」确认可用

### 5.2 支持的 LLM 端点

任何 OpenAI 兼容的 API 端点都可以使用：

| 服务商 | API 端点 |
|--------|---------|
| OpenAI | `https://api.openai.com/v1` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Ollama 本地 | `http://localhost:11434/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

---

## 六、评测报告解读

### 6.1 指标分数含义

| 分数范围 | 含义 |
|----------|------|
| 0.9 ~ 1.0 | 优秀 |
| 0.7 ~ 0.9 | 良好 |
| 0.5 ~ 0.7 | 有改进空间 |
| < 0.5 | 需要重点关注 |

### 6.2 逐条明细的 reason 字段

这是 LLM 对该条数据给出的评分依据。例如：

> Faithfulness = 0.67，reason: "回答中'多线程无法真正利用多核CPU'的表述在上下文中仅有间接支持"

这告诉你具体哪个陈述有问题，可以针对性地改进检索策略或 prompt。

### 6.3 按状态筛选

在报告明细 Tab 的下拉筛选中选择：
- **全部**：查看所有行
- **通过**：只看通过的行
- **不通过**：重点分析失败原因
- **错误**：查看评测过程中出错的行

---

## 七、常见问题

### Q: 评测一直在 running 状态不结束？

检查后端日志（终端输出），可能原因：
- LLM API 响应慢或超时 → 检查网络和 API Key
- 数据格式有误 → 检查数据集字段是否匹配场景要求

### Q: 报告中所有指标都是 None？

说明评测引擎在调用指标时出错。检查：
- 数据集字段名是否正确（如 `retrieved_contexts` 而非 `contexts`）
- LLM 配置是否有效（先测试连通性）

### Q: CSV 导入后 retrieved_contexts 字段显示为字符串？

CSV 中的 JSON 数组需要双引号转义：`"[""文本1"", ""文本2""]"`。推荐使用 JSON 格式导入复杂数据。

### Q: 如何更换 API Key？

1. 编辑 `backend/.env` 中的 `LLM_API_KEY`
2. 重启后端
3. 或者在界面的「LLM 配置」中编辑，填入新的 API Key 保存

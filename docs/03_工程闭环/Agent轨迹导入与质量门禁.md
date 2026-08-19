# Agent 轨迹导入与质量门禁

这条链路把 Agent 的黑盒执行结果统一为平台可评测的数据行：

```text
Agent events / trajectory
    -> normalize_agent_trace
    -> dataset row
    -> trajectory metrics + endpoint metrics
    -> report compare
    -> Trace Lint + LLM 轨迹指标
    -> Bad Case 分类 / 回归集
    -> quality gate: passed / blocked
```

## Trace 合同

导入接口接受 `POST /api/datasets/{dataset_id}/agent-traces/import`，请求体为：

```json
{
  "traces": [
    {
      "trace_id": "optional-id",
      "input": "查询订单 ORD-001",
      "output": "订单已发货",
      "available_tools": [
        {"name": "query_order", "description": "查询订单状态"}
      ],
      "events": [
        {
          "type": "assistant",
          "content": "我先查询订单。",
          "tool_calls": [
            {"id": "call-1", "name": "query_order", "arguments": {"order_id": "ORD-001"}}
          ]
        },
        {
          "type": "tool",
          "tool_call_id": "call-1",
          "name": "query_order",
          "content": "{\"status\":\"shipped\"}"
        },
        {"type": "assistant", "content": "订单已发货"}
      ],
      "reference": "查询订单并告知状态",
      "reference_tool_calls": [
        {"name": "query_order", "args": {"order_id": "ORD-001"}}
      ]
    }
  ]
}
```

规范化后会得到 `user_input`、`response`、`tool_calls`、`agent_trajectory`、`available_tools` 等通用字段。导入会先全量校验，再一次性写入；重复轨迹按规范化内容去重，并递增数据集版本。

## 轨迹指标

Agent 预设现在包含：

- `trajectory_faithfulness`：工具返回与后续推理/回答是否一致；
- `error_recovery`：工具报错后是否重试、切换工具或采用替代路径；没有错误时按“无需恢复”通过，但在理由中标明该样本不证明恢复能力；
- `tool_selection_rationality`：结合可用工具集判断当前选择是否合理。
- `trace_lint`：不调用 LLM，检查工具注册、参数 Schema、重复/失败调用、步数上限和高风险工具确认。

## Tool Registry

工具目录接口为 `/api/tool-registry`，每个工具保存名称、描述、JSON Schema、风险等级、是否有副作用、幂等要求和超时时间。创建评测任务时会冻结启用工具的快照，并参与评测指纹。

也可以直接调用 `POST /api/tool-registry/lint` 对单条轨迹做确定性检查。这样可以把“模型判断工具是否合理”和“平台先检查工具调用是否合规”分成两层。

## Bad Case 回流

- `GET /api/reports/{eval_id}/badcases`：按失败指标自动聚合 retrieval / generation / tool_selection / tool_arguments / trace_faithfulness / error_recovery / trace_lint / endpoint_error 分类；
- `PATCH /api/reports/{eval_id}/rows/{row_id}/badcase`：人工修正分类；
- `POST /api/reports/{eval_id}/badcases/regression-dataset`：把失败样本复制为新数据集，并保留分类和来源评测 ID。

这形成：失败样本 -> 分类 -> 人工修正 -> 回归集 -> 重新评测的最小闭环。

被测接口返回轨迹时，在接口目标的响应映射中配置 `agent_trajectory_path` 和 `available_tools_path`，平台会把事件列表转换为同一轨迹格式后再评分。

## 质量门禁

`POST /api/reports/{eval_id}/quality-gate` 接受基线评测 ID 和规则，例如：

```json
{
  "baseline_eval_id": 12,
  "require_identical_fingerprint": true,
  "maximum_new_failures": 0,
  "maximum_new_errors": 0,
  "maximum_cost_increase_cny": 0.01,
  "maximum_latency_p95_increase_ms": 500,
  "metric_rules": [
    {"metric": "trajectory_faithfulness", "minimum_mean_score": 0.8},
    {"metric": "error_recovery", "minimum_mean_delta": 0}
  ]
}
```

默认要求当前与基线的评测指纹一致；指纹未知或变化时会阻断，避免把“换了数据/指标/裁判”误判成 Agent 质量提升。响应包含 `passed`、`status`、逐条 `violations` 和完整对比结果，并返回成本与行级 P95 延迟变化，可阻断“质量上升但成本/延迟失控”的本地回归。

import React, { useState } from 'react';
import { Drawer, Typography, Tag, Divider, Table, Button, Tooltip } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;

interface MetricHelpInfo {
  name: string;
  displayName: string;
  category: string;
  compareSubject: string;
  compareMethod: string;
  scoreRange: string;
  requiredFields: string[];
  example: string;
}

const METRIC_HELP: Record<string, MetricHelpInfo> = {
  faithfulness: {
    name: 'faithfulness',
    displayName: '忠实度 (Faithfulness)',
    category: '生成可信度',
    compareSubject: 'response（模型回答）  vs  retrieved_contexts（检索到的上下文）',
    compareMethod: '1. 平台原生 Judge 阅读 response 和 retrieved_contexts\n2. 判断回答中的关键事实是否能从检索上下文得到直接或合理支持\n3. 按支持程度给出 0~1 分数\n4. 同时返回具体理由，指出支撑点或扣分点',
    scoreRange: '0.0 ~ 1.0（1.0 表示回答完全基于检索内容，无幻觉）',
    requiredFields: ['user_input', 'response', 'retrieved_contexts'],
    example: '回答说"Python由Guido在1991年创建"，如果检索内容里没有提到1991年，这个陈述就判为不忠实，拉低分数。',
  },
  context_recall: {
    name: 'context_recall',
    displayName: '上下文召回率 (Context Recall)',
    category: '检索质量',
    compareSubject: 'retrieved_contexts（检索到的上下文）  vs  reference（标准参考答案）',
    compareMethod: '1. 平台原生 Judge 阅读 reference 中的关键要点\n2. 检查 retrieved_contexts 是否覆盖这些要点和必要依据\n3. 覆盖越完整分数越高，遗漏关键依据会扣分\n4. reason 会说明哪些要点被覆盖或遗漏',
    scoreRange: '0.0 ~ 1.0（1.0 表示检索内容完全覆盖了标准答案的所有要点）',
    requiredFields: ['user_input', 'retrieved_contexts', 'reference'],
    example: '标准答案有3个要点，检索到的文档覆盖了2个 → 分数 = 2/3 ≈ 0.67',
  },
  context_precision: {
    name: 'context_precision',
    displayName: '上下文精确度 (Context Precision)',
    category: '检索质量',
    compareSubject: 'retrieved_contexts（检索到的上下文）  vs  reference（标准参考答案）',
    compareMethod: '1. 平台原生 Judge 逐一检查检索上下文是否与用户问题和参考答案相关\n2. 无关、重复、噪声或无法支撑回答的片段会扣分\n3. 分数反映召回内容的整体有用性\n4. reason 会指出有用片段和噪声片段',
    scoreRange: '0.0 ~ 1.0（1.0 表示检索到的全是有用信息，且排序正确）',
    requiredFields: ['user_input', 'retrieved_contexts', 'reference'],
    example: '检索到3段文档，第1段有用，第2段无关，第3段有用 → 因为无关内容排在中间，精确度会低于1.0',
  },
  answer_relevancy: {
    name: 'answer_relevancy',
    displayName: '回答相关性 (Answer Relevancy)',
    category: '回答质量',
    compareSubject: 'response（模型回答）  vs  user_input（用户问题）',
    compareMethod: '1. 平台原生 Judge 阅读用户问题和模型回答\n2. 判断回答是否直接回应问题，是否跑题、泛泛而谈或只回答了局部\n3. 按切题程度给出 0~1 分数\n4. reason 会说明回答贴合或偏离问题的地方',
    scoreRange: '0.0 ~ 1.0（1.0 表示回答完全切题，没有跑题内容）',
    requiredFields: ['user_input', 'response'],
    example: '用户问"Python的GIL是什么"，如果回答里大篇幅讲了Python的历史而非GIL，相关性就会低。',
  },
  factual_correctness: {
    name: 'factual_correctness',
    displayName: '事实正确性 (Factual Correctness)',
    category: '生成可信度',
    compareSubject: 'response（模型回答）  vs  reference（标准参考答案）',
    compareMethod: '1. 平台原生 Judge 对比 response 与 reference 中的事实陈述\n2. 事实冲突、过度绝对化、遗漏关键限定条件都会扣分\n3. 按事实一致程度给出 0~1 分数\n4. reason 会指出事实一致点、冲突点或遗漏点',
    scoreRange: '0.0 ~ 1.0（1.0 表示回答在事实层面与标准答案完全一致）',
    requiredFields: ['response', 'reference'],
    example: '回答说"GIL限制了所有类型的并行"，标准答案说"GIL限制CPU密集型并行" → 回答过于绝对，事实正确性扣分。',
  },
  answer_completeness: {
    name: 'answer_completeness',
    displayName: '答案完整性 (Answer Completeness)',
    category: '回答质量',
    compareSubject: 'response（模型回答）  vs  reference（标准参考答案）',
    compareMethod: '1. 平台原生 Judge 将 reference 拆解成必须覆盖的关键要点\n2. 检查 response 是否覆盖这些要点，以及是否遗漏关键条件、例外或步骤\n3. 按完整性给出 0~1 分数\n4. reason 会列出覆盖点和遗漏点\n\n它更关注“有没有答全”，与 factual_correctness 的“事实是否说对”互补。',
    scoreRange: '0.0 ~ 1.0（1.0 表示回答覆盖标准答案的全部关键要点）',
    requiredFields: ['user_input', 'response', 'reference'],
    example: '标准答案要求说明“GIL 影响 CPU 密集型线程，但 IO 密集型仍可能受益”。回答只说“GIL 限制并行” → 事实大致没错，但遗漏 IO 场景，完整性会扣分。',
  },
  retrieval_hit_rate: {
    name: 'retrieval_hit_rate',
    displayName: '召回命中率 (HitRate@K)',
    category: '检索单测扩展',
    compareSubject: 'retrieved_context_ids（实际召回文档ID）  vs  reference_context_ids（期望文档ID）',
    compareMethod: '1. 取 Top-K 的 retrieved_context_ids\n2. 判断其中是否至少命中 1 个 reference_context_ids\n3. 命中返回 1，未命中返回 0\n4. 多条样本求平均后得到整体召回成功率\n\n这是确定性代码指标，不依赖 LLM 评判。',
    scoreRange: '0 或 1；报告平均值为 0.0 ~ 1.0（越高表示检索越容易召回正确文档）',
    requiredFields: ['retrieved_context_ids', 'reference_context_ids'],
    example: '期望文档=[doc_1, doc_3]，Top-3 召回=[doc_7, doc_3, doc_9] → 命中 doc_3，HitRate@3 = 1。',
  },
  retrieval_mrr: {
    name: 'retrieval_mrr',
    displayName: '检索排序质量 (MRR)',
    category: '检索单测扩展',
    compareSubject: 'retrieved_context_ids（实际召回文档ID及排序）  vs  reference_context_ids（期望文档ID）',
    compareMethod: '1. 按顺序扫描 retrieved_context_ids\n2. 找到第一个命中的 reference_context_ids\n3. 分数 = 1 / 命中位置排名\n4. 如果没有命中，则分数为 0\n\n它不仅看是否召回，还看正确文档是否排在前面。',
    scoreRange: '0.0 ~ 1.0（1.0 表示第 1 位就是正确文档；0 表示完全未命中）',
    requiredFields: ['retrieved_context_ids', 'reference_context_ids'],
    example: '期望文档=[doc_9]，召回排序=[doc_1, doc_9, doc_3] → 第 2 位命中，MRR = 1/2 = 0.5。',
  },
  tool_call_accuracy: {
    name: 'tool_call_accuracy',
    displayName: '工具调用准确度 (Tool Call Accuracy)',
    category: 'Agent',
    compareSubject: 'AI 实际调用的工具  vs  reference_tool_calls（期望调用的工具）',
    compareMethod: '1. 后端从 conversation 中提取 AI 消息里的 tool_calls（实际调用）\n2. 与 reference_tool_calls（期望调用）逐一匹配\n3. 工具名称和参数完全一致才算命中\n4. 分数 = 完全匹配的期望调用数 / 期望调用总数\n\n这是确定性代码指标，不依赖 LLM。',
    scoreRange: '0.0 ~ 1.0（1.0 表示期望工具调用全部被正确执行）',
    requiredFields: ['user_input（conversation格式）', 'reference_tool_calls'],
    example: '期望调用 query_order(id="001") → 实际调用 query_order(id="001") → 1.0\n期望调用 query_order(id="001") → 实际调用 query_order(id="002") → 参数不匹配，扣分',
  },
  agent_goal_accuracy: {
    name: 'agent_goal_accuracy',
    displayName: '目标达成度 (Agent Goal Accuracy)',
    category: 'Agent',
    compareSubject: '对话的最终结果  vs  reference（期望目标）',
    compareMethod: '1. 平台原生 Judge 阅读完整 Agent 轨迹，包括用户请求、工具调用、工具返回和最终回复\n2. 判断最终状态是否达成 reference 描述的业务目标\n3. 按目标达成程度给出 0~1 分数\n4. reason 会说明达成、部分达成或失败的依据',
    scoreRange: '0.0 ~ 1.0（越高表示越充分达成用户目标）',
    requiredFields: ['user_input（conversation格式）', 'reference'],
    example: '用户要求"帮我订明天的会议室"，对话结束时会议室订好了 → 1\n对话结束时 Agent 说"稍后处理" → 0',
  },
  topic_adherence: {
    name: 'topic_adherence',
    displayName: '话题遵守度 (Topic Adherence)',
    category: '多轮对话',
    compareSubject: '对话中涉及的话题  vs  reference_topics（允许的话题列表）',
    compareMethod: '1. 平台原生 Judge 阅读完整多轮对话\n2. 判断讨论内容是否围绕 reference_topics 指定的话题范围\n3. 明显越界、跑题或忽略当前轮次主题会扣分\n4. reason 会指出哪些轮次遵守或偏离话题',
    scoreRange: '0.0 ~ 1.0（1.0 表示对话完全围绕允许的话题）',
    requiredFields: ['user_input（conversation格式）', 'reference_topics'],
    example: '允许话题=[退货,售后]，对话中AI回答了退货问题(正常)也推荐了商品(超范围) → 精确度下降',
  },
  harmfulness: {
    name: 'harmfulness',
    displayName: '有害性检测 (Harmfulness)',
    category: '通用',
    compareSubject: 'response（模型回答）独立评判',
    compareMethod: '1. 平台原生 Judge 阅读 response 或对话内容\n2. 根据定义判断是否有害、欺骗性或可能造成伤害\n3. 返回 0 或 1，并给出具体判断理由',
    scoreRange: '0 或 1（0 = 无害/安全，1 = 有害）',
    requiredFields: ['response'],
    example: '回答教用户如何绕过安全限制 → 1（有害）\n正常回答技术问题 → 0（无害）',
  },
  coherence: {
    name: 'coherence',
    displayName: '连贯性 (Coherence)',
    category: '通用',
    compareSubject: 'response（模型回答）独立评判',
    compareMethod: '1. 平台原生 Judge 阅读 response 或完整 conversation\n2. 判断回答是否逻辑通顺、条理清晰、前后不矛盾、上下文衔接自然\n3. 返回 0 或 1，并给出具体判断理由',
    scoreRange: '0 或 1（0 = 不连贯/混乱，1 = 连贯/有条理）',
    requiredFields: ['response'],
    example: '回答前后矛盾、逻辑跳跃 → 0\n回答层次分明、步步推进 → 1',
  },
};

export const MetricHelpIcon: React.FC<{ metricName: string; onClick: () => void }> = ({ metricName, onClick }) => {
  const info = METRIC_HELP[metricName];
  const tip = info ? `点击查看评分标准和计算方式：${info.compareSubject}` : '点击查看指标详情';
  return (
    <Tooltip title={tip} placement="top">
      <QuestionCircleOutlined
        style={{ color: '#1677ff', cursor: 'pointer', marginLeft: 4, fontSize: 13 }}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
      />
    </Tooltip>
  );
};

export const MetricHelpDrawer: React.FC<{
  open: boolean;
  metricName: string | null;
  onClose: () => void;
}> = ({ open, metricName, onClose }) => {
  const info = metricName ? METRIC_HELP[metricName] : null;

  return (
    <Drawer
      title={info ? info.displayName : '指标说明'}
      open={open}
      onClose={onClose}
      width={600}
    >
      {info ? (
        <Typography>
          <Tag color="blue" style={{ marginBottom: 12 }}>{info.category}</Tag>

          <Title level={5}>比较对象</Title>
          <Paragraph>
            <Text code style={{ fontSize: 14, lineHeight: 2 }}>{info.compareSubject}</Text>
          </Paragraph>

          <Divider />

          <Title level={5}>计算方法</Title>
          <Paragraph>
            <pre style={{
              background: '#f5f5f5', padding: 12, borderRadius: 6,
              fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap',
            }}>
              {info.compareMethod}
            </pre>
          </Paragraph>

          <Divider />

          <Title level={5}>分数范围</Title>
          <Paragraph><Text strong>{info.scoreRange}</Text></Paragraph>

          <Divider />

          <Title level={5}>所需数据字段</Title>
          <Paragraph>
            {info.requiredFields.map((f) => (
              <Tag key={f} color="geekblue" style={{ marginBottom: 4 }}>{f}</Tag>
            ))}
          </Paragraph>

          <Divider />

          <Title level={5}>通俗示例</Title>
          <Paragraph>
            <pre style={{
              background: '#fffbe6', padding: 12, borderRadius: 6, border: '1px solid #ffe58f',
              fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap',
            }}>
              {info.example}
            </pre>
          </Paragraph>
        </Typography>
      ) : (
        <Paragraph>
          该指标为自定义指标，暂无内置说明。请参考创建指标时填写的 definition 或 prompt 了解评判逻辑。
        </Paragraph>
      )}
    </Drawer>
  );
};

export const MetricHelpButton: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  const allMetrics = Object.values(METRIC_HELP);

  const columns = [
    {
      title: '指标',
      dataIndex: 'displayName',
      key: 'displayName',
      width: 220,
      render: (v: string, record: MetricHelpInfo) => (
        <Button type="link" style={{ padding: 0 }} onClick={() => setSelectedMetric(record.name)}>
          {v}
        </Button>
      ),
    },
    { title: '类别', dataIndex: 'category', key: 'category', width: 80, render: (v: string) => <Tag color="blue">{v}</Tag> },
    { title: '比较对象', dataIndex: 'compareSubject', key: 'compareSubject', ellipsis: true },
    { title: '分数', dataIndex: 'scoreRange', key: 'scoreRange', width: 120, ellipsis: true },
  ];

  return (
    <>
      <Button
        type="link"
        icon={<QuestionCircleOutlined />}
        onClick={() => setOpen(true)}
        style={{ fontSize: 13 }}
      >
        指标说明
      </Button>
      <Drawer title="评测指标说明" open={open} onClose={() => { setOpen(false); setSelectedMetric(null); }} width={750}>
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          点击指标名称查看详细的计算方法和示例。LLM 类指标由平台原生 Judge 调用评测用 LLM 打分并返回理由；确定性指标由后端代码直接计算。
        </Paragraph>
        <Table
          rowKey="name"
          columns={columns}
          dataSource={allMetrics}
          pagination={false}
          size="small"
        />
        <MetricHelpDrawer open={!!selectedMetric} metricName={selectedMetric} onClose={() => setSelectedMetric(null)} />
      </Drawer>
    </>
  );
};

export default METRIC_HELP;

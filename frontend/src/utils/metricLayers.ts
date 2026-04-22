export interface MetricLayer {
  key: string;
  name: string;
  description: string;
  color: string;
  headerBg: string;
  subHeaderBg: string;
  headerBorder: string;
  headerText: string;
}

export interface MetricInfo {
  name: string;
  displayName: string;
  shortName: string;
  layerKey: string;
  short: string;
  subjects: string;
  defaultEnabled?: boolean;
}

export const METRIC_LAYERS: MetricLayer[] = [
  {
    key: 'retrieval_quality',
    name: '检索质量',
    description: '检索是否覆盖标准答案要点，以及召回内容是否足够有用。',
    color: 'blue',
    headerBg: '#e6f4ff',
    subHeaderBg: '#f0f8ff',
    headerBorder: '#1677ff',
    headerText: '#0958d9',
  },
  {
    key: 'generation_trust',
    name: '生成可信度',
    description: '回答是否基于检索内容，事实是否与标准答案一致。',
    color: 'green',
    headerBg: '#f6ffed',
    subHeaderBg: '#fbfff5',
    headerBorder: '#52c41a',
    headerText: '#237804',
  },
  {
    key: 'answer_quality',
    name: '回答质量',
    description: '回答是否切题、是否完整覆盖参考答案的关键要点。',
    color: 'purple',
    headerBg: '#f9f0ff',
    subHeaderBg: '#fcf7ff',
    headerBorder: '#722ed1',
    headerText: '#531dab',
  },
  {
    key: 'retrieval_unit',
    name: '检索单测扩展',
    description: '基于文档/分片 ID 的确定性检索测试，适合单独验证召回和排序。',
    color: 'cyan',
    headerBg: '#e6fffb',
    subHeaderBg: '#f0fffc',
    headerBorder: '#13c2c2',
    headerText: '#006d75',
  },
  {
    key: 'agent',
    name: 'Agent 能力',
    description: '工具调用是否正确、任务目标是否完成，聚焦 Agent 的行动链路。',
    color: 'orange',
    headerBg: '#fff7e6',
    subHeaderBg: '#fffbf0',
    headerBorder: '#fa8c16',
    headerText: '#ad4e00',
  },
  {
    key: 'conversation',
    name: '对话质量',
    description: '多轮对话是否围绕业务话题、是否前后连贯，聚焦对话体验。',
    color: 'magenta',
    headerBg: '#fff0f6',
    subHeaderBg: '#fff7fb',
    headerBorder: '#eb2f96',
    headerText: '#c41d7f',
  },
  {
    key: 'general',
    name: '通用/自定义',
    description: '安全性、连贯性或业务自定义指标。',
    color: 'default',
    headerBg: '#f5f5f5',
    subHeaderBg: '#fafafa',
    headerBorder: '#8c8c8c',
    headerText: '#434343',
  },
];

export const METRIC_INFO: Record<string, MetricInfo> = {
  context_recall: {
    name: 'context_recall',
    displayName: '上下文召回率 (Context Recall)',
    shortName: '上下文召回率',
    layerKey: 'retrieval_quality',
    short: '检索上下文 vs 标准答案',
    subjects: '原生 Judge 指标：检查标准答案(reference)里的关键要点，是否都能在被测系统检索出的上下文(retrieved_contexts)里找到，并返回具体理由。',
    defaultEnabled: true,
  },
  context_precision: {
    name: 'context_precision',
    displayName: '上下文精确度 (Context Precision)',
    shortName: '上下文精确度',
    layerKey: 'retrieval_quality',
    short: '检索上下文 vs 标准答案',
    subjects: '原生 Judge 指标：检查被测系统检索到的上下文(retrieved_contexts)中，有多少内容对用户问题和参考答案真正有用，并返回具体理由。',
    defaultEnabled: true,
  },
  faithfulness: {
    name: 'faithfulness',
    displayName: '忠实度 (Faithfulness)',
    shortName: '忠实度',
    layerKey: 'generation_trust',
    short: 'AI回答 vs 检索上下文',
    subjects: '原生 Judge 指标：检查被测系统回答(response)中的关键陈述，是否都能从检索上下文(retrieved_contexts)中得到支撑，并返回具体理由。',
    defaultEnabled: true,
  },
  factual_correctness: {
    name: 'factual_correctness',
    displayName: '事实正确性 (Factual Correctness)',
    shortName: '事实正确性',
    layerKey: 'generation_trust',
    short: 'AI回答 vs 标准答案',
    subjects: '原生 Judge 指标：检查被测系统回答(response)在事实层面是否与标准答案(reference)一致，并指出事实冲突或遗漏。',
    defaultEnabled: true,
  },
  answer_relevancy: {
    name: 'answer_relevancy',
    displayName: '回答相关性 (Answer Relevancy)',
    shortName: '回答相关性',
    layerKey: 'answer_quality',
    short: 'AI回答 vs 用户问题',
    subjects: '原生 Judge 指标：检查被测系统回答(response)是否切题，是否直接回应用户问题(user_input)，并说明跑题或覆盖不足之处。',
    defaultEnabled: true,
  },
  answer_completeness: {
    name: 'answer_completeness',
    displayName: '答案完整性 (Answer Completeness)',
    shortName: '答案完整性',
    layerKey: 'answer_quality',
    short: 'AI回答 vs 标准答案',
    subjects: '原生 Judge 指标：检查被测系统回答(response)是否完整覆盖标准答案(reference)中的关键要点、条件和步骤，并说明遗漏点。',
    defaultEnabled: true,
  },
  retrieval_hit_rate: {
    name: 'retrieval_hit_rate',
    displayName: '召回命中率 (HitRate@K)',
    shortName: 'HitRate@K',
    layerKey: 'retrieval_unit',
    short: '召回文档ID vs 期望文档ID',
    subjects: '确定性代码指标：Top-K 召回结果(retrieved_context_ids)是否命中任一标准文档/分片 ID(reference_context_ids)。适合检索单测，不依赖 LLM。',
    defaultEnabled: false,
  },
  retrieval_mrr: {
    name: 'retrieval_mrr',
    displayName: '检索排序质量 (MRR)',
    shortName: 'MRR',
    layerKey: 'retrieval_unit',
    short: '召回排序 vs 期望文档ID',
    subjects: '确定性代码指标：第一个正确文档/分片 ID 在召回列表里的倒数排名。越靠前分数越高，未命中为 0。适合检索排序单测。',
    defaultEnabled: false,
  },
  tool_call_accuracy: {
    name: 'tool_call_accuracy',
    displayName: '工具调用准确度 (Tool Call Accuracy)',
    shortName: '工具调用准确度',
    layerKey: 'agent',
    short: '实际工具调用 vs 期望工具调用',
    subjects: '确定性代码指标：从 conversation 中提取 AI 的 tool_calls，与 reference_tool_calls 比较工具名称和参数，给出匹配数量说明。',
  },
  agent_goal_accuracy: {
    name: 'agent_goal_accuracy',
    displayName: '目标达成度 (Agent Goal Accuracy)',
    shortName: '目标达成度',
    layerKey: 'agent',
    short: '对话结果 vs 期望目标',
    subjects: '原生 Judge 指标：阅读完整 Agent 轨迹，判断最终状态是否达成 reference 中定义的业务目标，并返回具体理由。',
  },
  topic_adherence: {
    name: 'topic_adherence',
    displayName: '话题遵守度 (Topic Adherence)',
    shortName: '话题遵守度',
    layerKey: 'conversation',
    short: '对话话题 vs 允许话题',
    subjects: '原生 Judge 指标：检查多轮对话是否围绕 reference_topics 中允许的话题展开，识别跑题或越界回答。',
  },
  harmfulness: {
    name: 'harmfulness',
    displayName: '有害性检测 (Harmfulness)',
    shortName: '有害性检测',
    layerKey: 'general',
    short: 'AI回答独立安全评判',
    subjects: '原生 Judge/自定义维度指标：检查回答或对话是否包含有害、欺骗性或可能造成伤害的内容，并说明判断依据。',
  },
  coherence: {
    name: 'coherence',
    displayName: '连贯性 (Coherence)',
    shortName: '连贯性',
    layerKey: 'conversation',
    short: 'AI回答独立连贯性评判',
    subjects: '原生 Judge/自定义维度指标：检查回答或对话是否逻辑通顺、前后不矛盾、上下文衔接自然。',
  },
};

export const getMetricInfo = (name: string): MetricInfo => (
  METRIC_INFO[name] || {
    name,
    displayName: name,
    shortName: name,
    layerKey: 'general',
    short: '',
    subjects: '自定义指标，请参考指标定义中的 prompt 或 definition。',
  }
);

export const getMetricLayer = (metricName: string): MetricLayer => {
  const info = getMetricInfo(metricName);
  return METRIC_LAYERS.find((layer) => layer.key === info.layerKey) || METRIC_LAYERS[METRIC_LAYERS.length - 1];
};

export const getMetricLayerIndex = (metricName: string) => {
  const layer = getMetricLayer(metricName);
  const index = METRIC_LAYERS.findIndex((item) => item.key === layer.key);
  return index >= 0 ? index : METRIC_LAYERS.length;
};

export const groupMetricNames = (metricNames: string[]) => {
  const uniqueNames = Array.from(new Set(metricNames));
  return METRIC_LAYERS.map((layer) => ({
    ...layer,
    metrics: uniqueNames.filter((name) => getMetricInfo(name).layerKey === layer.key),
  })).filter((group) => group.metrics.length > 0);
};

export const isMetricAvailableForScene = (metric: { category?: string; name?: string }, sceneType: string) => {
  if (metric.category === sceneType || metric.category === 'custom' || metric.category === 'general') {
    return true;
  }
  return sceneType === 'rag' && (metric.category === 'rag_retrieval' || getMetricInfo(metric.name || '').layerKey === 'retrieval_unit');
};

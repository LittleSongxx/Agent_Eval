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
    name: 'Agent 任务',
    description: '任务目标是否完成，最终结果是否符合用户意图。',
    color: 'orange',
    headerBg: '#fff7e6',
    subHeaderBg: '#fffbf0',
    headerBorder: '#fa8c16',
    headerText: '#ad4e00',
  },
  {
    key: 'agent_tooling',
    name: 'Agent 工具',
    description: '工具是否选对，参数是否正确。',
    color: 'gold',
    headerBg: '#fffbe6',
    subHeaderBg: '#fffdf0',
    headerBorder: '#faad14',
    headerText: '#ad6800',
  },
  {
    key: 'agent_process',
    name: 'Agent 过程',
    description: '执行步骤是否高效，是否存在多余或缺失步骤。',
    color: 'volcano',
    headerBg: '#fff2e8',
    subHeaderBg: '#fff7f0',
    headerBorder: '#fa541c',
    headerText: '#ad2102',
  },
  {
    key: 'conversation',
    name: '对话相关性',
    description: '每轮回复是否回应当前用户问题，是否围绕业务话题。',
    color: 'magenta',
    headerBg: '#fff0f6',
    subHeaderBg: '#fff7fb',
    headerBorder: '#eb2f96',
    headerText: '#c41d7f',
  },
  {
    key: 'conversation_outcome',
    name: '对话完成度',
    description: '整段对话是否满足用户需求并完成期望结果。',
    color: 'lime',
    headerBg: '#fcffe6',
    subHeaderBg: '#fefff0',
    headerBorder: '#a0d911',
    headerText: '#5b8c00',
  },
  {
    key: 'conversation_memory',
    name: '上下文记忆',
    description: '是否记住用户已提供的事实、偏好、限制和上下文。',
    color: 'geekblue',
    headerBg: '#f0f5ff',
    subHeaderBg: '#f5f8ff',
    headerBorder: '#2f54eb',
    headerText: '#1d39c4',
  },
  {
    key: 'conversation_role',
    name: '角色一致性',
    description: '是否遵守角色、职责边界和语气要求。',
    color: 'red',
    headerBg: '#fff1f0',
    subHeaderBg: '#fff7f5',
    headerBorder: '#f5222d',
    headerText: '#a8071a',
  },
  {
    key: 'conversation_grounding',
    name: '多轮可信度',
    description: '多轮回复是否基于检索上下文或给定资料。',
    color: 'green',
    headerBg: '#f6ffed',
    subHeaderBg: '#fbfff5',
    headerBorder: '#52c41a',
    headerText: '#237804',
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
  contextual_relevancy: {
    name: 'contextual_relevancy',
    displayName: '上下文相关性 (Contextual Relevancy)',
    shortName: '上下文相关性',
    layerKey: 'retrieval_quality',
    short: '检索上下文 vs 用户问题',
    subjects: '原生 Judge 指标：检查检索上下文整体是否与用户问题直接相关，识别无关、泛化、只沾边或无法帮助回答的片段。',
    defaultEnabled: false,
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
    displayName: '工具正确性 (Tool Correctness)',
    shortName: '工具正确性',
    layerKey: 'agent_tooling',
    short: '实际工具调用 vs 期望工具调用',
    subjects: '确定性代码指标：从 conversation 中提取 AI 的 tool_calls，与 reference_tool_calls 比较工具名称和参数，给出匹配数量说明。',
  },
  task_completion: {
    name: 'task_completion',
    displayName: '任务完成度 (Task Completion)',
    shortName: '任务完成度',
    layerKey: 'agent',
    short: 'Agent 轨迹 vs 期望任务',
    subjects: '原生 Judge 指标：阅读完整 Agent 轨迹，判断最终状态是否完成 reference 中描述的任务闭环。',
  },
  agent_goal_accuracy: {
    name: 'agent_goal_accuracy',
    displayName: '目标准确度 (Goal Accuracy)',
    shortName: '目标准确度',
    layerKey: 'agent',
    short: '对话结果 vs 期望目标',
    subjects: '原生 Judge 指标：阅读完整 Agent 轨迹，判断最终状态是否达成 reference 中定义的业务目标，并返回具体理由。',
  },
  argument_correctness: {
    name: 'argument_correctness',
    displayName: '参数正确性 (Argument Correctness)',
    shortName: '参数正确性',
    layerKey: 'agent_tooling',
    short: '实际参数 vs 期望参数',
    subjects: '确定性代码指标：在工具名称匹配的前提下，计算实际工具参数与 reference_tool_calls 中期望参数的匹配度。',
  },
  step_efficiency: {
    name: 'step_efficiency',
    displayName: '步骤效率 (Step Efficiency)',
    shortName: '步骤效率',
    layerKey: 'agent_process',
    short: '实际步骤数 vs 期望步骤数',
    subjects: '确定性代码指标：比较实际工具调用步骤数和期望步骤数，惩罚多余、重复或缺失的工具步骤。',
  },
  topic_adherence: {
    name: 'topic_adherence',
    displayName: '话题遵守度 (Topic Adherence)',
    shortName: '话题遵守度',
    layerKey: 'conversation',
    short: '对话话题 vs 允许话题',
    subjects: '原生 Judge 指标：检查多轮对话是否围绕 reference_topics 中允许的话题展开，识别跑题或越界回答。',
  },
  turn_relevancy: {
    name: 'turn_relevancy',
    displayName: '轮次相关性 (Turn Relevancy)',
    shortName: '轮次相关性',
    layerKey: 'conversation',
    short: '每轮 AI 回复 vs 当前用户输入',
    subjects: '原生 Judge 指标：逐轮检查 AI 回复是否回应当前用户问题，是否忽略追问、答非所问或错误带入上下文。',
  },
  conversation_completeness: {
    name: 'conversation_completeness',
    displayName: '对话完成度 (Conversation Completeness)',
    shortName: '对话完成度',
    layerKey: 'conversation_outcome',
    short: '完整对话 vs 期望结果',
    subjects: '原生 Judge 指标：判断整段对话是否满足用户需求并完成 reference 描述的对话目标。',
  },
  knowledge_retention: {
    name: 'knowledge_retention',
    displayName: '上下文记忆 (Knowledge Retention)',
    shortName: '上下文记忆',
    layerKey: 'conversation_memory',
    short: '后续回复 vs 已给定事实',
    subjects: '原生 Judge 指标：检查 AI 是否记住用户前面给出的事实、偏好、订单号、时间和限制条件。',
  },
  role_adherence: {
    name: 'role_adherence',
    displayName: '角色遵守度 (Role Adherence)',
    shortName: '角色遵守度',
    layerKey: 'conversation_role',
    short: '完整对话 vs 指定角色',
    subjects: '原生 Judge 指标：检查 AI 是否遵守 reference_role 中定义的角色、职责边界和语气要求。',
  },
  turn_faithfulness: {
    name: 'turn_faithfulness',
    displayName: '轮次忠实度 (Turn Faithfulness)',
    shortName: '轮次忠实度',
    layerKey: 'conversation_grounding',
    short: '多轮回复 vs 检索上下文',
    subjects: '原生 Judge 指标：检查多轮回复是否基于 retrieved_contexts 或给定资料，识别跨轮幻觉和上下文冲突。',
    defaultEnabled: false,
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
    layerKey: 'general',
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

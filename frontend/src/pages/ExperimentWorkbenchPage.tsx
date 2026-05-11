import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Input,
  Modal,
  Progress,
  Radio,
  Row,
  Select,
  Space,
  Spin,
  Steps,
  Tag,
  Timeline,
  Typography,
  message,
} from 'antd';
import {
  ApiOutlined,
  BarChartOutlined,
  CodeOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  FormOutlined,
  PlayCircleOutlined,
  SafetyCertificateOutlined,
  SwapOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { Dataset, EndpointTarget, EvalScenario, EvalTask, LLMConfig, MetricDefinition } from '../types';
import * as api from '../services/api';
import DatasetCreateModal from '../components/resource/DatasetCreateModal';
import EndpointTargetModal from '../components/resource/EndpointTargetModal';
import LLMConfigModal from '../components/resource/LLMConfigModal';
import ScenarioQuickCreateModal from '../components/resource/ScenarioQuickCreateModal';
import DatasetRowsPanel from '../components/resource/DatasetRowsPanel';

const { Title, Text } = Typography;
const { TextArea } = Input;

const DEFAULT_TEST_INPUT = '请用三句话介绍一下 DeepSeek，并说明它适合做哪些 AI 应用测试。';
const WORKBENCH_DRAFT_STORAGE_KEY = 'ai-eval-platform.workbenchDraft';

const statusColorMap: Record<string, string> = {
  pending: 'blue',
  running: 'orange',
  completed: 'green',
  failed: 'red',
  cancelled: 'default',
};

const statusLabelMap: Record<string, string> = {
  pending: '等待中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const metricRequiredFields: Record<string, string[]> = {
  faithfulness: ['response', 'retrieved_contexts'],
  context_recall: ['retrieved_contexts', 'reference'],
  context_precision: ['user_input', 'retrieved_contexts'],
  contextual_relevancy: ['user_input', 'retrieved_contexts'],
  factual_correctness: ['response', 'reference'],
  answer_relevancy: ['user_input', 'response'],
  answer_completeness: ['response', 'reference'],
  retrieval_hit_rate: ['retrieved_context_ids', 'reference_context_ids'],
  retrieval_mrr: ['retrieved_context_ids', 'reference_context_ids'],
  task_completion: ['user_input', 'reference'],
  tool_call_accuracy: ['user_input', 'reference_tool_calls'],
  argument_correctness: ['user_input', 'reference_tool_calls'],
  step_efficiency: ['user_input', 'reference_tool_calls'],
  agent_goal_accuracy: ['user_input', 'reference'],
  topic_adherence: ['user_input', 'reference_topics'],
  turn_relevancy: ['user_input'],
  conversation_completeness: ['user_input', 'reference'],
  knowledge_retention: ['user_input'],
  role_adherence: ['user_input', 'reference_role'],
  turn_faithfulness: ['user_input', 'retrieved_contexts'],
};

type EvaluationMode = 'offline' | 'endpoint';
type ResourceSource = 'existing' | 'create';
type TestStatus = 'idle' | 'success' | 'failed';

interface ExperimentDraft {
  evaluation_mode: EvaluationMode;
  dataset_id?: number;
  endpoint_target_id?: number;
  scenario_id?: number;
  llm_config_id?: number;
  task_id?: number;
}

const loadWorkbenchDraft = (): ExperimentDraft => {
  if (typeof window === 'undefined') return { evaluation_mode: 'offline' };
  try {
    const raw = window.localStorage.getItem(WORKBENCH_DRAFT_STORAGE_KEY);
    if (!raw) return { evaluation_mode: 'offline' };
    const parsed = JSON.parse(raw);
    return {
      evaluation_mode: parsed.evaluation_mode === 'endpoint' ? 'endpoint' : 'offline',
      dataset_id: typeof parsed.dataset_id === 'number' ? parsed.dataset_id : undefined,
      endpoint_target_id: typeof parsed.endpoint_target_id === 'number' ? parsed.endpoint_target_id : undefined,
      scenario_id: typeof parsed.scenario_id === 'number' ? parsed.scenario_id : undefined,
      llm_config_id: typeof parsed.llm_config_id === 'number' ? parsed.llm_config_id : undefined,
      task_id: typeof parsed.task_id === 'number' ? parsed.task_id : undefined,
    };
  } catch {
    return { evaluation_mode: 'offline' };
  }
};

const ExperimentWorkbenchPage: React.FC = () => {
  const navigate = useNavigate();
  const [draft, setDraft] = useState<ExperimentDraft>(() => loadWorkbenchDraft());
  const [currentStep, setCurrentStep] = useState(0);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [endpointTargets, setEndpointTargets] = useState<EndpointTarget[]>([]);
  const [scenarios, setScenarios] = useState<EvalScenario[]>([]);
  const [llmConfigs, setLLMConfigs] = useState<LLMConfig[]>([]);
  const [metrics, setMetrics] = useState<MetricDefinition[]>([]);
  const [tasks, setTasks] = useState<EvalTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [debugging, setDebugging] = useState(false);
  const [debugResult, setDebugResult] = useState<any>(null);
  const [debugModalOpen, setDebugModalOpen] = useState(false);
  const [testingEndpoint, setTestingEndpoint] = useState(false);
  const [testingLLM, setTestingLLM] = useState(false);
  const [endpointTestInput, setEndpointTestInput] = useState(DEFAULT_TEST_INPUT);
  const [endpointTestStatus, setEndpointTestStatus] = useState<TestStatus>('idle');
  const [endpointTestResult, setEndpointTestResult] = useState<any>(null);
  const [llmTestStatus, setLlmTestStatus] = useState<TestStatus>('idle');
  const [llmTestMessage, setLlmTestMessage] = useState('');
  const [datasetSource, setDatasetSource] = useState<ResourceSource>('existing');
  const [endpointSource, setEndpointSource] = useState<ResourceSource>('existing');
  const [scenarioSource, setScenarioSource] = useState<ResourceSource>('existing');
  const [llmSource, setLlmSource] = useState<ResourceSource>('existing');
  const [datasetModalOpen, setDatasetModalOpen] = useState(false);
  const [endpointModalOpen, setEndpointModalOpen] = useState(false);
  const [scenarioModalOpen, setScenarioModalOpen] = useState(false);
  const [llmModalOpen, setLlmModalOpen] = useState(false);

  const selectedDataset = datasets.find((item) => item.id === draft.dataset_id);
  const selectedEndpoint = endpointTargets.find((item) => item.id === draft.endpoint_target_id);
  const selectedScenario = scenarios.find((item) => item.id === draft.scenario_id);
  const selectedLLM = llmConfigs.find((item) => item.id === draft.llm_config_id);
  const currentTask = tasks.find((item) => item.id === draft.task_id);
  const activeTasks = tasks.filter((item) => item.status === 'pending' || item.status === 'running');

  const handleDatasetUpdated = useCallback((nextDataset: Dataset) => {
    setDatasets((prev) => prev.map((item) => (item.id === nextDataset.id ? nextDataset : item)));
  }, []);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [ds, et, sc, lc, metricData, taskData] = await Promise.all([
        api.listDatasets(),
        api.listEndpointTargets(),
        api.listScenarios(),
        api.listLLMConfigs(),
        api.listMetrics(),
        api.listEvaluations(),
      ]);
      setDatasets(Array.isArray(ds) ? ds : []);
      setEndpointTargets(Array.isArray(et) ? et : []);
      setScenarios(Array.isArray(sc) ? sc : []);
      setLLMConfigs(Array.isArray(lc) ? lc : []);
      setMetrics(Array.isArray(metricData) ? metricData : []);
      const normalizedTasks = Array.isArray(taskData) ? taskData : taskData?.items || [];
      setTasks(normalizedTasks);
    } catch {
      message.error('加载工作台数据失败');
    } finally {
      setLoading(false);
    }
  };

  const openScenarioCreateModal = async () => {
    await fetchAll();
    setScenarioModalOpen(true);
  };

  useEffect(() => {
    fetchAll();
  }, []);

  useEffect(() => {
    window.localStorage.setItem(WORKBENCH_DRAFT_STORAGE_KEY, JSON.stringify(draft));
  }, [draft]);

  useEffect(() => {
    if (!draft.task_id) return;
    const timer = setInterval(async () => {
      try {
        const data = await api.listEvaluations();
        const nextTasks = Array.isArray(data) ? data : data?.items || [];
        setTasks(nextTasks);
      } catch {
        // Keep the current screen stable; the user can refresh manually.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [draft.task_id]);

  const endpointProducedFields = useMemo(() => {
    const mapping = selectedEndpoint?.response_mapping || {};
    const produced = new Set<string>();
    if (mapping.response_path) produced.add('response');
    if (mapping.retrieved_contexts_path) produced.add('retrieved_contexts');
    if (mapping.retrieved_context_ids_path) produced.add('retrieved_context_ids');
    if (mapping.tool_calls_path) produced.add('tool_calls');
    return produced;
  }, [selectedEndpoint]);

  const compatibility = useMemo(() => {
    if (!selectedDataset || !selectedScenario) {
      return { ok: true, missingFields: [] as string[], message: '选择数据集和评测标准后会自动检查字段。' };
    }
    if (selectedDataset.sample_type !== selectedScenario.sample_type) {
      return {
        ok: false,
        missingFields: [] as string[],
        message: `样本类型不匹配：数据集是 ${selectedDataset.sample_type}，场景需要 ${selectedScenario.sample_type}。`,
      };
    }
    const datasetFields = new Set((selectedDataset.field_schema || []).map((field) => field.name));
    const requiredFields = new Set<string>();
    (selectedScenario.metrics || []).forEach((scenarioMetric) => {
      const metricName = scenarioMetric.metric_definition?.name;
      (metricName ? metricRequiredFields[metricName] || [] : []).forEach((field) => requiredFields.add(field));
    });
    const missingFields = Array.from(requiredFields).filter((field) => {
      if (datasetFields.has(field)) return false;
      return draft.evaluation_mode === 'endpoint' ? !endpointProducedFields.has(field) : true;
    });
    if (missingFields.length > 0) {
      return {
        ok: false,
        missingFields,
        message: `数据集缺少当前场景必需字段：${missingFields.join(', ')}。`,
      };
    }
    return { ok: true, missingFields: [] as string[], message: '数据集字段满足当前场景的核心指标要求。' };
  }, [draft.evaluation_mode, endpointProducedFields, selectedDataset, selectedScenario]);

  const steps = useMemo(() => {
    const base = [
      { key: 'mode', title: '选择评测模式', icon: <ExperimentOutlined /> },
      { key: 'dataset', title: '准备测试数据', icon: <DatabaseOutlined /> },
    ];
    if (draft.evaluation_mode === 'endpoint') {
      base.push({ key: 'endpoint', title: '配置被测接口', icon: <ApiOutlined /> });
    }
    base.push(
      { key: 'scenario', title: '选择评测标准', icon: <SafetyCertificateOutlined /> },
      { key: 'llm', title: '配置 Judge LLM', icon: <FormOutlined /> },
      { key: 'run', title: '执行评测', icon: <PlayCircleOutlined /> },
      { key: 'report', title: '查看报告', icon: <BarChartOutlined /> },
      { key: 'review', title: '人工复核/盲测', icon: <SwapOutlined /> },
    );
    return base;
  }, [draft.evaluation_mode]);

  useEffect(() => {
    if (currentStep > steps.length - 1) {
      setCurrentStep(steps.length - 1);
    }
  }, [currentStep, steps.length]);

  const stepIndex = (key: string) => steps.findIndex((item) => item.key === key);
  const isTaskDone = currentTask?.status === 'completed';
  const customPromptCount = (selectedScenario?.metrics || []).filter((item) => item.prompt_override).length;
  const selectedScenarioMetricCount = selectedScenario?.metrics?.length || 0;
  const datasetReady = Boolean(selectedDataset && selectedDataset.row_count > 0);
  const endpointExtractedFields = endpointTestResult?.extracted_fields || {};
  const endpointExtractedResponse = endpointExtractedFields.response;
  const endpointResponseParsed = endpointExtractedResponse !== null
    && endpointExtractedResponse !== undefined
    && endpointExtractedResponse !== '';

  const executionBlocks = useMemo(() => {
    const blocks: string[] = [];
    if (!selectedDataset) {
      blocks.push('未选择测试数据集');
    } else if (selectedDataset.row_count <= 0) {
      blocks.push('数据集还没有数据行，请先导入数据或添加一行');
    }
    if (draft.evaluation_mode === 'endpoint' && !selectedEndpoint) {
      blocks.push('接口实时评测需要选择被测接口');
    }
    if (!selectedScenario) {
      blocks.push('未选择评测场景/评测标准');
    } else if ((selectedScenario.metrics || []).length === 0) {
      blocks.push('当前评测场景没有指标，请先添加至少 1 个指标');
    }
    if (selectedDataset && selectedScenario && !compatibility.ok) {
      blocks.push(compatibility.message);
    }
    if (!selectedLLM) {
      blocks.push('未选择 Judge LLM 配置');
    }
    return blocks;
  }, [compatibility, draft.evaluation_mode, selectedDataset, selectedEndpoint, selectedLLM, selectedScenario]);

  const executionWarnings = useMemo(() => {
    const warnings: string[] = [];
    if (draft.evaluation_mode === 'endpoint' && selectedEndpoint && endpointTestStatus !== 'success') {
      warnings.push('建议先试跑被测接口，确认请求模板、鉴权和字段映射都可用');
    }
    if (draft.evaluation_mode === 'endpoint' && endpointTestStatus === 'success' && !endpointResponseParsed) {
      warnings.push('接口试跑成功但没有解析出 response 字段，评测可能无法评分');
    }
    if (selectedLLM && llmTestStatus !== 'success') {
      warnings.push('建议先测试 Judge LLM 连接，避免任务开始后才失败');
    }
    return warnings;
  }, [draft.evaluation_mode, endpointResponseParsed, endpointTestStatus, llmTestStatus, selectedEndpoint, selectedLLM]);

  const canProceed = (key: string) => {
    if (key === 'mode') return true;
    if (key === 'dataset') return datasetReady;
    if (key === 'endpoint') return draft.evaluation_mode !== 'endpoint' || Boolean(draft.endpoint_target_id);
    if (key === 'scenario') return Boolean(draft.scenario_id) && selectedScenarioMetricCount > 0 && compatibility.ok;
    if (key === 'llm') return Boolean(draft.llm_config_id);
    if (key === 'run') return Boolean(draft.task_id);
    if (key === 'report') return Boolean(isTaskDone);
    if (key === 'review') return Boolean(isTaskDone);
    return false;
  };

  const getStepStatus = (index: number): 'wait' | 'process' | 'finish' | 'error' => {
    const key = steps[index]?.key;
    if (index === currentStep) {
      if (key === 'dataset' && selectedDataset && selectedDataset.row_count <= 0) return 'error';
      if (key === 'scenario' && draft.dataset_id && draft.scenario_id && !compatibility.ok) return 'error';
      if (key === 'scenario' && selectedScenario && selectedScenarioMetricCount === 0) return 'error';
      return 'process';
    }
    if (index < currentStep && canProceed(key)) return 'finish';
    return 'wait';
  };

  const goNext = () => {
    if (currentStep < steps.length - 1) setCurrentStep(currentStep + 1);
  };

  const goPrev = () => {
    if (currentStep > 0) setCurrentStep(currentStep - 1);
  };

  const handleModeChange = (mode: EvaluationMode) => {
    setDraft((prev) => ({
      evaluation_mode: mode,
      dataset_id: prev.dataset_id,
      scenario_id: prev.scenario_id,
      llm_config_id: prev.llm_config_id,
      endpoint_target_id: mode === 'endpoint' ? prev.endpoint_target_id : undefined,
    }));
  };

  const handleTestEndpoint = async () => {
    if (!selectedEndpoint) {
      message.warning('请先选择被测接口');
      return;
    }
    setTestingEndpoint(true);
    setEndpointTestStatus('idle');
    setEndpointTestResult(null);
    try {
      const result = await api.testEndpointTarget(selectedEndpoint.id, {
        row_data: { user_input: endpointTestInput || selectedEndpoint.default_test_input || DEFAULT_TEST_INPUT },
      });
      setEndpointTestResult(result);
      if (result.success) {
        setEndpointTestStatus('success');
        message.success('接口试跑成功');
      } else {
        setEndpointTestStatus('failed');
        message.error(`接口试跑失败: ${result.message || '未知错误'}`);
      }
    } catch {
      setEndpointTestStatus('failed');
      message.error('接口试跑请求失败');
    } finally {
      setTestingEndpoint(false);
    }
  };

  const handleTestLLM = async () => {
    if (!selectedLLM) {
      message.warning('请先选择 LLM 配置');
      return;
    }
    setTestingLLM(true);
    setLlmTestStatus('idle');
    setLlmTestMessage('');
    try {
      const result = await api.testLLMConfig(selectedLLM.id);
      setLlmTestMessage(result.message || '');
      if (result.success) {
        setLlmTestStatus('success');
        message.success('LLM 连接测试成功');
      } else {
        setLlmTestStatus('failed');
        message.error(result.message || 'LLM 连接测试失败');
      }
    } catch {
      setLlmTestStatus('failed');
      setLlmTestMessage('LLM 连接测试请求失败');
      message.error('LLM 连接测试请求失败');
    } finally {
      setTestingLLM(false);
    }
  };

  const buildEvaluationPayload = () => {
    const timestamp = new Date().toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
    const payload: Record<string, any> = {
      name: `工作台评测 ${timestamp}`,
      evaluation_mode: draft.evaluation_mode,
      dataset_id: draft.dataset_id,
      scenario_id: draft.scenario_id,
      llm_config_id: draft.llm_config_id,
      result_save_mode: 'task_only',
    };
    if (draft.evaluation_mode === 'endpoint' && selectedEndpoint) {
      payload.endpoint_target_id = selectedEndpoint.id;
      payload.target_config = {
        endpoint_url: selectedEndpoint.endpoint_url,
        transport_mode: selectedEndpoint.transport_mode || 'json',
        authorization: selectedEndpoint.authorization || '',
        extra_headers: selectedEndpoint.extra_headers || '{}',
        request_body_template: selectedEndpoint.request_body_template,
      };
      payload.response_mapping = selectedEndpoint.response_mapping || {};
    }
    return payload;
  };

  const handleDebugEvaluation = async () => {
    if (executionBlocks.length > 0) {
      message.warning(executionBlocks[0]);
      return;
    }
    setDebugging(true);
    try {
      const result = await api.debugEvaluation(buildEvaluationPayload());
      setDebugResult(result);
      setDebugModalOpen(true);
      if (result.success) {
        message.success('评测流程验证通过');
      } else {
        message.warning('评测流程验证完成，但存在错误或风险');
      }
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      message.error(typeof detail === 'string' ? detail : '评测流程验证失败');
    } finally {
      setDebugging(false);
    }
  };

  const createEvaluation = async () => {
    if (executionBlocks.length > 0) {
      message.warning(executionBlocks[0]);
      return;
    }

    const payload = buildEvaluationPayload();

    setCreating(true);
    try {
      const created = await api.createEvaluation(payload);
      setDraft((prev) => ({ ...prev, task_id: created.id }));
      setTasks((prev) => [created, ...prev.filter((item) => item.id !== created.id)]);
      message.success('评测任务已创建');
      setCurrentStep(stepIndex('run'));
    } catch {
      message.error('创建评测任务失败');
    } finally {
      setCreating(false);
    }
  };

  const taskPercent = currentTask?.total_rows
    ? Math.min(100, Math.round(((currentTask.completed_rows || 0) / currentTask.total_rows) * 100))
    : Math.round((currentTask?.progress || 0) * 100);

  const renderStepContent = () => {
    const key = steps[currentStep]?.key;
    if (key === 'mode') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>选择评测模式</Title>
          <Radio.Group
            value={draft.evaluation_mode}
            onChange={(event) => handleModeChange(event.target.value)}
            optionType="button"
            buttonStyle="solid"
            options={[
              { label: '已有结果评测', value: 'offline' },
              { label: '接口实时评测', value: 'endpoint' },
            ]}
          />
          <Alert
            type="info"
            showIcon
            message={draft.evaluation_mode === 'endpoint' ? '接口实时评测' : '已有结果评测'}
            description={draft.evaluation_mode === 'endpoint'
              ? '测试同学只需要准备输入集、被测接口和评估标准，平台会先调用业务接口，再复用现有 Judge 指标评分。'
              : '数据集中已经包含 response、retrieved_contexts 等被测输出字段，平台直接对已有结果评分。'}
          />
        </Space>
      );
    }

    if (key === 'dataset') {
      const fields = selectedDataset?.field_schema || [];
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>准备测试数据</Title>
          <Radio.Group
            value={datasetSource}
            onChange={(event) => setDatasetSource(event.target.value)}
            optionType="button"
            buttonStyle="solid"
            options={[
              { label: '使用已有数据集', value: 'existing' },
              { label: '新建数据集', value: 'create' },
            ]}
          />
          {datasetSource === 'existing' ? (
            <Select
              showSearch
              optionFilterProp="label"
              style={{ width: 520, maxWidth: '100%' }}
              placeholder="选择已有数据集"
              value={draft.dataset_id}
              onChange={(value) => setDraft((prev) => ({ ...prev, dataset_id: value }))}
              options={datasets.map((dataset) => ({
                label: `${dataset.name} · ${dataset.sample_type} · ${dataset.row_count} 条`,
                value: dataset.id,
              }))}
            />
          ) : (
            <Card size="small">
              <Space direction="vertical" size={12}>
                <Text>先创建数据集字段结构；创建成功后继续在本节点导入数据或添加一行。</Text>
                <Button type="primary" onClick={() => setDatasetModalOpen(true)}>打开新建数据集弹窗</Button>
              </Space>
            </Card>
          )}
          {datasets.length === 0 && (
            <Alert
              type="warning"
              showIcon
              message="还没有数据集"
              description="可在当前节点新建数据集，表单与数据管理页面保持一致。"
            />
          )}
          <Alert
            type="info"
            showIcon
            message={draft.evaluation_mode === 'endpoint' ? '接口实时评测的数据准备要求' : '已有结果评测的数据准备要求'}
            description={draft.evaluation_mode === 'endpoint'
              ? '至少需要有输入字段，例如 user_input；如果指标会参考答案或业务标准，建议同时准备 reference、rubrics 等字段。接口返回的 response 可由被测接口映射产生。'
              : '数据集中需要已经包含被测输出，例如 response；如果指标依赖召回、工具或参考答案，还需要 retrieved_contexts、tool_calls、reference 等字段。'}
          />
          {selectedDataset && (
            <>
              <Card size="small">
                <Descriptions size="small" column={2}>
                  <Descriptions.Item label="数据集">{selectedDataset.name}</Descriptions.Item>
                  <Descriptions.Item label="样本类型">{selectedDataset.sample_type}</Descriptions.Item>
                  <Descriptions.Item label="样本数">
                    <Tag color={selectedDataset.row_count > 0 ? 'green' : 'orange'}>
                      {selectedDataset.row_count}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="字段">
                    <Space wrap>
                      {fields.map((field) => <Tag key={field.name}>{field.name}</Tag>)}
                    </Space>
                  </Descriptions.Item>
                </Descriptions>
              </Card>
              {selectedDataset.row_count <= 0 && (
                <Alert
                  type="warning"
                  showIcon
                  message="数据集还没有数据行"
                  description="只有创建字段结构还不能开始评测，请在下面导入文件或手动添加一行测试数据。"
                />
              )}
              <DatasetRowsPanel
                dataset={selectedDataset}
                compact
                onDatasetUpdated={handleDatasetUpdated}
              />
            </>
          )}
        </Space>
      );
    }

    if (key === 'endpoint') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>配置被测接口</Title>
          <Radio.Group
            value={endpointSource}
            onChange={(event) => setEndpointSource(event.target.value)}
            optionType="button"
            buttonStyle="solid"
            options={[
              { label: '使用已有接口', value: 'existing' },
              { label: '新建被测接口', value: 'create' },
            ]}
          />
          {endpointSource === 'existing' ? (
            <Select
              showSearch
              optionFilterProp="label"
              style={{ width: 560, maxWidth: '100%' }}
              placeholder="选择已保存被测接口"
              value={draft.endpoint_target_id}
              onChange={(value) => {
                setDraft((prev) => ({ ...prev, endpoint_target_id: value }));
                const target = endpointTargets.find((item) => item.id === value);
                setEndpointTestInput(target?.default_test_input || DEFAULT_TEST_INPUT);
                setEndpointTestStatus('idle');
                setEndpointTestResult(null);
              }}
              options={endpointTargets.map((target) => ({
                label: `${target.name} · ${target.transport_mode.toUpperCase()} · ${target.response_mapping?.response_path || '-'}`,
                value: target.id,
              }))}
            />
          ) : (
            <Card size="small">
              <Space direction="vertical" size={12}>
                <Text>新建接口时请配置请求模板、鉴权和响应字段映射；创建成功后会自动选中并进入试跑。</Text>
                <Button type="primary" onClick={() => setEndpointModalOpen(true)}>打开新建被测接口弹窗</Button>
              </Space>
            </Card>
          )}
          {endpointTargets.length === 0 && (
            <Alert
              type="warning"
              showIcon
              message="还没有被测接口"
              description="可在当前节点新建被测接口，表单与被测接口管理页面保持一致。"
            />
          )}
          {selectedEndpoint && (
            <>
              <Card size="small">
                <Descriptions size="small" column={2}>
                  <Descriptions.Item label="接口名称">{selectedEndpoint.name}</Descriptions.Item>
                  <Descriptions.Item label="返回方式">{selectedEndpoint.transport_mode}</Descriptions.Item>
                  <Descriptions.Item label="接口地址" span={2}>{selectedEndpoint.endpoint_url}</Descriptions.Item>
                  <Descriptions.Item label="回答字段">{selectedEndpoint.response_mapping?.response_path || '-'}</Descriptions.Item>
                  <Descriptions.Item label="上下文字段">{selectedEndpoint.response_mapping?.retrieved_contexts_path || '-'}</Descriptions.Item>
                </Descriptions>
              </Card>
              <TextArea
                value={endpointTestInput}
                onChange={(event) => setEndpointTestInput(event.target.value)}
                autoSize={{ minRows: 3, maxRows: 5 }}
                placeholder="输入一条试跑问题"
              />
              <Button loading={testingEndpoint} icon={<ApiOutlined />} onClick={handleTestEndpoint}>
                试跑接口
              </Button>
              {endpointTestStatus !== 'idle' && (
                <Alert
                  type={endpointTestStatus === 'success' ? (endpointResponseParsed ? 'success' : 'warning') : 'error'}
                  showIcon
                  message={endpointTestStatus === 'success'
                    ? endpointResponseParsed
                      ? '接口试跑成功，已解析出 response'
                      : '接口试跑成功，但未解析出 response'
                    : '接口试跑失败'}
                  description={endpointTestStatus === 'success'
                    ? '下面展示本次试跑的请求体、原始响应摘要和字段映射结果。'
                    : endpointTestResult?.message || '请检查接口地址、鉴权、请求模板或网络可达性。'}
                />
              )}
              {endpointTestResult && (
                <Card size="small" title="试跑结果">
                  <Space direction="vertical" style={{ width: '100%' }} size={12}>
                    <div>
                      <Text strong>请求体</Text>
                      <pre style={{ maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                        {JSON.stringify(endpointTestResult.request_body || {}, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <Text strong>解析字段</Text>
                      <pre style={{ maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                        {JSON.stringify(endpointTestResult.extracted_fields || {}, null, 2)}
                      </pre>
                    </div>
                    {endpointTestResult.raw_response && (
                      <div>
                        <Text strong>原始响应摘要</Text>
                        <pre style={{ maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                          {typeof endpointTestResult.raw_response === 'string'
                            ? endpointTestResult.raw_response.slice(0, 2000)
                            : JSON.stringify(endpointTestResult.raw_response, null, 2).slice(0, 2000)}
                        </pre>
                      </div>
                    )}
                    {endpointTestResult.mapping_errors && Object.keys(endpointTestResult.mapping_errors).length > 0 && (
                      <Alert
                        type="warning"
                        showIcon
                        message="字段映射存在问题"
                        description={JSON.stringify(endpointTestResult.mapping_errors)}
                      />
                    )}
                  </Space>
                </Card>
              )}
            </>
          )}
        </Space>
      );
    }

    if (key === 'scenario') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>选择评测标准</Title>
          <Radio.Group
            value={scenarioSource}
            onChange={(event) => setScenarioSource(event.target.value)}
            optionType="button"
            buttonStyle="solid"
            options={[
              { label: '使用已有场景', value: 'existing' },
              { label: '新建评测场景', value: 'create' },
            ]}
          />
          {scenarioSource === 'existing' ? (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Space wrap>
                <Select
                  showSearch
                  optionFilterProp="label"
                  style={{ width: 560, maxWidth: '100%' }}
                  placeholder="选择评测场景"
                  value={draft.scenario_id}
                  onChange={(value) => setDraft((prev) => ({ ...prev, scenario_id: value }))}
                  options={scenarios.map((scenario) => ({
                    label: `${scenario.name} · ${scenario.scene_type} · ${scenario.metrics?.length || 0} 个指标`,
                    value: scenario.id,
                  }))}
                />
                <Button onClick={fetchAll}>刷新资源</Button>
              </Space>
              <Alert
                type="info"
                showIcon
                message="这里选择的是评测场景，不是单个指标"
                description={`新建指标后，需要在“新建评测场景”或“场景管理”中把该指标加入场景，执行评测时才会使用。当前已加载 ${metrics.length} 个指标定义、${scenarios.length} 个场景。`}
              />
            </Space>
          ) : (
            <Card size="small">
              <Space direction="vertical" size={12}>
                <Alert
                  type="info"
                  showIcon
                  message="这里是快速创建"
                  description="可以完成场景名称、样本类型和指标选择；复杂规则、场景级提示词覆盖可到场景管理继续编辑。"
                />
                <Button type="primary" onClick={openScenarioCreateModal}>打开新建评测场景弹窗</Button>
              </Space>
            </Card>
          )}
          {scenarios.length === 0 && (
            <Alert
              type="warning"
              showIcon
              message="还没有评测场景"
              description="可在当前节点快速创建评测场景，复杂配置可到场景管理继续编辑。"
            />
          )}
          {selectedScenario && (
            <>
              <Card size="small">
                <Descriptions size="small" column={2}>
                  <Descriptions.Item label="场景">{selectedScenario.name}</Descriptions.Item>
                  <Descriptions.Item label="样本类型">{selectedScenario.sample_type}</Descriptions.Item>
                  <Descriptions.Item label="指标数">
                    <Tag color={selectedScenarioMetricCount > 0 ? 'green' : 'orange'}>{selectedScenarioMetricCount}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="场景覆盖规则">{customPromptCount}</Descriptions.Item>
                </Descriptions>
                <Divider style={{ margin: '12px 0' }} />
                <Space wrap>
                  {(selectedScenario.metrics || []).map((metric) => (
                    <Tag key={metric.metric_definition_id} color={metric.prompt_override ? 'processing' : 'default'}>
                      {metric.metric_definition?.display_name || metric.metric_definition?.name || metric.metric_definition_id}
                      {metric.pass_threshold != null ? ` ≥${metric.pass_threshold}` : ''}
                    </Tag>
                  ))}
                </Space>
              </Card>
              <Alert
                type={compatibility.ok ? 'success' : 'warning'}
                showIcon
                message={compatibility.ok ? '数据集与场景字段匹配' : '数据集与场景不匹配'}
                description={compatibility.message}
              />
              {selectedScenarioMetricCount === 0 && (
                <Alert
                  type="warning"
                  showIcon
                  message="当前场景没有指标"
                  description="评测标准必须至少包含 1 个指标，否则无法执行评分。"
                />
              )}
            </>
          )}
        </Space>
      );
    }

    if (key === 'llm') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>配置 Judge LLM</Title>
          <Radio.Group
            value={llmSource}
            onChange={(event) => setLlmSource(event.target.value)}
            optionType="button"
            buttonStyle="solid"
            options={[
              { label: '使用已有配置', value: 'existing' },
              { label: '新建 LLM 配置', value: 'create' },
            ]}
          />
          {llmSource === 'existing' ? (
            <Select
              showSearch
              optionFilterProp="label"
              style={{ width: 560, maxWidth: '100%' }}
              placeholder="选择评测裁判使用的 LLM 配置"
              value={draft.llm_config_id}
              onChange={(value) => {
                setDraft((prev) => ({ ...prev, llm_config_id: value }));
                setLlmTestStatus('idle');
                setLlmTestMessage('');
              }}
              options={llmConfigs.map((config) => ({
                label: `${config.name} · ${config.model_name}`,
                value: config.id,
              }))}
            />
          ) : (
            <Card size="small">
              <Space direction="vertical" size={12}>
                <Text>新建 Judge LLM 后会自动选中；建议继续测试连接，确认 API Key、模型名和地址可用。</Text>
                <Button type="primary" onClick={() => setLlmModalOpen(true)}>打开新建 LLM 配置弹窗</Button>
              </Space>
            </Card>
          )}
          {llmConfigs.length === 0 && (
            <Alert
              type="warning"
              showIcon
              message="还没有 LLM 配置"
              description="可在当前节点新建 LLM 配置，表单与 LLM 配置管理页面保持一致。"
            />
          )}
          {selectedLLM && (
            <>
              <Card size="small">
                <Descriptions size="small" column={2}>
                  <Descriptions.Item label="配置名">{selectedLLM.name}</Descriptions.Item>
                  <Descriptions.Item label="模型">{selectedLLM.model_name}</Descriptions.Item>
                  <Descriptions.Item label="服务商">{selectedLLM.provider || '-'}</Descriptions.Item>
                  <Descriptions.Item label="默认配置">{selectedLLM.is_default ? '是' : '否'}</Descriptions.Item>
                </Descriptions>
              </Card>
              <Space wrap>
                <Button loading={testingLLM} onClick={handleTestLLM}>测试 LLM 连接</Button>
                {llmTestStatus === 'success' && <Tag color="green">连接成功</Tag>}
                {llmTestStatus === 'failed' && <Tag color="red">连接失败</Tag>}
                {llmTestMessage && <Text type={llmTestStatus === 'failed' ? 'danger' : 'secondary'}>{llmTestMessage}</Text>}
              </Space>
            </>
          )}
        </Space>
      );
    }

    if (key === 'run') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>执行评测</Title>
          <Alert
            type={executionBlocks.length === 0 ? 'success' : 'warning'}
            showIcon
            message={executionBlocks.length === 0 ? '阻塞项已通过' : '还有阻塞项需要处理'}
            description={executionBlocks.length === 0
              ? '点击开始评测后会创建一条评测任务，并在后台执行。'
              : executionBlocks.join('；')}
          />
          {executionWarnings.length > 0 && (
            <Alert
              type="info"
              showIcon
              message="建议检查项"
              description={executionWarnings.join('；')}
            />
          )}
          <Card size="small" title="实验配置确认">
            <Descriptions size="small" column={2}>
              <Descriptions.Item label="模式">{draft.evaluation_mode === 'endpoint' ? '接口实时评测' : '已有结果评测'}</Descriptions.Item>
              <Descriptions.Item label="数据集">
                {selectedDataset ? `${selectedDataset.name}（${selectedDataset.row_count} 条）` : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="被测接口">{draft.evaluation_mode === 'endpoint' ? selectedEndpoint?.name || '-' : '不需要'}</Descriptions.Item>
              <Descriptions.Item label="评测标准">
                {selectedScenario ? `${selectedScenario.name}（${selectedScenarioMetricCount} 个指标）` : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="Judge LLM">{selectedLLM?.name || '-'}</Descriptions.Item>
              <Descriptions.Item label="结果保存">只存任务结果</Descriptions.Item>
            </Descriptions>
          </Card>
          <Space>
            <Button
              icon={<CodeOutlined />}
              loading={debugging}
              disabled={executionBlocks.length > 0}
              onClick={handleDebugEvaluation}
            >
              评测流程验证
            </Button>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              loading={creating}
              disabled={executionBlocks.length > 0}
              onClick={createEvaluation}
            >
              开始评测
            </Button>
            <Button icon={<CodeOutlined />} onClick={() => navigate('/evaluations')}>查看历史任务</Button>
          </Space>
          {currentTask && (
            <Card size="small" title="当前任务进度">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Space>
                  <Text strong>{currentTask.name}</Text>
                  <Tag color={statusColorMap[currentTask.status] || 'default'}>
                    {statusLabelMap[currentTask.status] || currentTask.status}
                  </Tag>
                  <Text type="secondary">
                    {currentTask.completed_rows || 0}/{currentTask.total_rows || 0}
                  </Text>
                </Space>
                <Progress
                  percent={taskPercent}
                  status={currentTask.status === 'failed' ? 'exception' : currentTask.status === 'completed' ? 'success' : 'active'}
                />
                {currentTask.error_message && <Text type="danger">{currentTask.error_message}</Text>}
              </Space>
            </Card>
          )}
        </Space>
      );
    }

    if (key === 'report') {
      return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Title level={4}>查看报告</Title>
          {isTaskDone ? (
            <Alert
              type="success"
              showIcon
              message="评测已完成，可以查看报告"
              description="报告中会区分样本通过率、指标通过率，并支持逐条明细和基线对比。"
            />
          ) : (
            <Alert
              type="info"
              showIcon
              message="评测完成后会在这里进入报告"
              description={currentTask ? `当前任务状态：${statusLabelMap[currentTask.status] || currentTask.status}` : '请先执行评测任务。'}
            />
          )}
          <Space>
            <Button type="primary" disabled={!isTaskDone || !draft.task_id} icon={<BarChartOutlined />} onClick={() => navigate(`/reports/${draft.task_id}`)}>
              查看本次报告
            </Button>
            <Button onClick={() => navigate('/reports')}>进入报告列表</Button>
          </Space>
        </Space>
      );
    }

    return (
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Title level={4}>人工复核 / 盲测</Title>
        <Alert
          type="info"
          showIcon
          message="自动评测后建议对关键失败样本做人工复核"
          description="如果需要比较两个模型或两个接口版本，可以进入人工盲测；如果只是确认失败原因，可以先在报告逐条明细中做人工复核。"
        />
        <Space>
          <Button disabled={!isTaskDone || !draft.task_id} icon={<FileSearchOutlined />} onClick={() => navigate(`/reports/${draft.task_id}`)}>
            去报告逐条复核
          </Button>
          <Button icon={<SwapOutlined />} onClick={() => navigate('/blind-tests')}>
            进入人工盲测
          </Button>
        </Space>
      </Space>
    );
  };

  const renderJsonBlock = (value: any) => (
    <pre
      style={{
        maxHeight: 280,
        overflow: 'auto',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        background: '#f5f5f5',
        borderRadius: 6,
        padding: 12,
        margin: 0,
      }}
    >
      {typeof value === 'string' ? value : JSON.stringify(value ?? {}, null, 2)}
    </pre>
  );

  return (
    <>
      <Spin spinning={loading}>
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <div>
            <Title level={3} style={{ marginBottom: 4 }}>评测实验工作台</Title>
            <Text type="secondary">按一次完整 AI 评测实验的顺序完成准备、执行、报告和人工复核。</Text>
          </div>
          {!currentTask && draft.task_id && (
            <Alert
              type="warning"
              showIcon
              message={`未找到上次工作台任务 #${draft.task_id}`}
              description="工作台已恢复上次草稿，但该任务没有出现在评测任务列表中。可以进入评测执行查看全部历史任务，或重新创建本次实验。"
              action={<Button size="small" onClick={() => navigate('/evaluations')}>去评测执行</Button>}
            />
          )}
          {!draft.task_id && activeTasks.length > 0 && (
            <Alert
              type="info"
              showIcon
              message={`当前有 ${activeTasks.length} 个评测任务正在执行或等待中`}
              description="这些任务来自后端任务列表，即使不是当前工作台草稿创建，也可以到评测执行查看进度和日志。"
              action={<Button size="small" onClick={() => navigate('/evaluations')}>查看任务</Button>}
            />
          )}

          <Row gutter={16} align="top">
            <Col xs={24} lg={7} xl={6}>
              <Card title="实验流程" bodyStyle={{ paddingBottom: 8 }}>
                <Steps
                  direction="vertical"
                  current={currentStep}
                  items={steps.map((step, index) => ({
                    title: step.title,
                    icon: step.icon,
                    status: getStepStatus(index),
                    description: index < currentStep && canProceed(step.key) ? '已完成' : undefined,
                  }))}
                  onChange={setCurrentStep}
                />
              </Card>
              <Card title="当前实验摘要" size="small" style={{ marginTop: 16 }}>
                <Timeline
                  items={[
                    { color: 'blue', children: draft.evaluation_mode === 'endpoint' ? '接口实时评测' : '已有结果评测' },
                    { color: datasetReady ? 'green' : draft.dataset_id ? 'orange' : 'gray', children: selectedDataset ? `${selectedDataset.name} · ${selectedDataset.row_count} 条` : '未选择数据集' },
                    ...(draft.evaluation_mode === 'endpoint' ? [{ color: draft.endpoint_target_id ? 'green' : 'gray', children: selectedEndpoint?.name || '未选择被测接口' }] : []),
                    { color: draft.scenario_id && selectedScenarioMetricCount > 0 && compatibility.ok ? 'green' : draft.scenario_id ? 'orange' : 'gray', children: selectedScenario?.name || '未选择评测标准' },
                    { color: draft.llm_config_id ? 'green' : 'gray', children: selectedLLM?.name || '未选择 Judge LLM' },
                    { color: isTaskDone ? 'green' : draft.task_id ? 'orange' : 'gray', children: currentTask?.name || '未创建任务' },
                  ]}
                />
              </Card>
            </Col>
            <Col xs={24} lg={17} xl={18}>
              <Card
                style={{ minHeight: 520 }}
                extra={
                  <Space>
                    <Button disabled={currentStep === 0} onClick={goPrev}>上一步</Button>
                    <Button
                      type="primary"
                      disabled={currentStep === steps.length - 1 || !canProceed(steps[currentStep]?.key || '')}
                      onClick={goNext}
                    >
                      下一步
                    </Button>
                  </Space>
                }
              >
                {renderStepContent()}
              </Card>
            </Col>
          </Row>
        </Space>
      </Spin>

      <Modal
        title="评测流程验证结果"
        open={debugModalOpen}
        onCancel={() => setDebugModalOpen(false)}
        footer={<Button type="primary" onClick={() => setDebugModalOpen(false)}>知道了</Button>}
        width={980}
      >
        {debugResult && (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Alert
              type={debugResult.success ? 'success' : 'warning'}
              showIcon
              message={debugResult.message || (debugResult.success ? '验证通过' : '验证存在风险')}
              description={
                <Space wrap>
                  <Tag color={debugResult.is_pass ? 'green' : 'red'}>
                    {debugResult.is_pass ? '样本通过' : '样本不通过'}
                  </Tag>
                  <Text type="secondary">耗时 {debugResult.execution_time_ms || 0} ms</Text>
                  <Text type="secondary">样本行 #{debugResult.dataset_row?.row_index ?? '-'}</Text>
                </Space>
              }
            />
            {debugResult.errors?.length > 0 && (
              <Alert type="error" showIcon message="错误" description={debugResult.errors.join('；')} />
            )}
            {debugResult.warnings?.length > 0 && (
              <Alert type="warning" showIcon message="风险提示" description={debugResult.warnings.join('；')} />
            )}
            <Card size="small" title="1. 样本输入">
              {renderJsonBlock(debugResult.dataset_row?.data || debugResult.row_data)}
            </Card>
            {debugResult.endpoint_trace && (
              <Card size="small" title="2. 被测接口调用结果">
                <Space direction="vertical" size={12} style={{ width: '100%' }}>
                  <Descriptions size="small" column={3}>
                    <Descriptions.Item label="状态">{debugResult.endpoint_trace.status || '-'}</Descriptions.Item>
                    <Descriptions.Item label="HTTP 状态码">{debugResult.endpoint_trace.status_code || '-'}</Descriptions.Item>
                    <Descriptions.Item label="耗时">{debugResult.endpoint_trace.latency_ms || '-'} ms</Descriptions.Item>
                  </Descriptions>
                  <Text strong>请求体</Text>
                  {renderJsonBlock(debugResult.endpoint_trace.request_body)}
                  <Text strong>映射字段</Text>
                  {renderJsonBlock(debugResult.endpoint_trace.extracted_fields)}
                  <Text strong>原始响应</Text>
                  {renderJsonBlock(debugResult.endpoint_trace.raw_response || debugResult.endpoint_trace.error)}
                </Space>
              </Card>
            )}
            <Card size="small" title="3. Judge Prompt 与返回结果">
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                {(debugResult.judge_traces || []).map((trace: any) => (
                  <Card key={trace.metric_name} size="small" title={trace.metric_display_name || trace.metric_name}>
                    <Space direction="vertical" size={12} style={{ width: '100%' }}>
                      {trace.error && <Alert type="error" showIcon message={trace.error} />}
                      {trace.warnings?.length > 0 && (
                        <Alert type="warning" showIcon message="Prompt/字段风险" description={trace.warnings.join('；')} />
                      )}
                      <Text strong>完整 Judge Prompt</Text>
                      {trace.prompt_messages ? renderJsonBlock(trace.prompt_messages) : <Text type="secondary">该指标不需要 Judge LLM Prompt</Text>}
                      <Text strong>Judge 原始返回</Text>
                      {trace.judge_raw_response ? renderJsonBlock(trace.judge_raw_response) : <Text type="secondary">无模型原始返回</Text>}
                      <Text strong>解析结果</Text>
                      {renderJsonBlock(trace.parsed_result)}
                    </Space>
                  </Card>
                ))}
              </Space>
            </Card>
            <Card size="small" title="4. 指标汇总">
              {renderJsonBlock(debugResult.metric_scores)}
            </Card>
          </Space>
        )}
      </Modal>

      <DatasetCreateModal
        open={datasetModalOpen}
        initialSampleType={selectedScenario?.sample_type || 'single_turn'}
        onCancel={() => setDatasetModalOpen(false)}
        onCreated={(dataset) => {
          setDatasets((prev) => [dataset, ...prev.filter((item) => item.id !== dataset.id)]);
          setDraft((prev) => ({ ...prev, dataset_id: dataset.id }));
          setDatasetSource('existing');
        }}
      />
      <EndpointTargetModal
        open={endpointModalOpen}
        onCancel={() => setEndpointModalOpen(false)}
        onSaved={(target) => {
          setEndpointTargets((prev) => [target, ...prev.filter((item) => item.id !== target.id)]);
          setDraft((prev) => ({ ...prev, endpoint_target_id: target.id }));
          setEndpointTestInput(target.default_test_input || DEFAULT_TEST_INPUT);
          setEndpointSource('existing');
          setEndpointTestStatus('idle');
          setEndpointTestResult(null);
        }}
      />
      <ScenarioQuickCreateModal
        open={scenarioModalOpen}
        metrics={metrics}
        initialSampleType={selectedDataset?.sample_type || 'single_turn'}
        onCancel={() => setScenarioModalOpen(false)}
        onCreated={(scenario) => {
          setScenarios((prev) => [scenario, ...prev.filter((item) => item.id !== scenario.id)]);
          setDraft((prev) => ({ ...prev, scenario_id: scenario.id }));
          setScenarioSource('existing');
        }}
      />
      <LLMConfigModal
        open={llmModalOpen}
        onCancel={() => setLlmModalOpen(false)}
        onSaved={(config) => {
          setLLMConfigs((prev) => [config, ...prev.filter((item) => item.id !== config.id)]);
          setDraft((prev) => ({ ...prev, llm_config_id: config.id }));
          setLlmSource('existing');
          setLlmTestStatus('idle');
          setLlmTestMessage('');
        }}
      />
    </>
  );
};

export default ExperimentWorkbenchPage;

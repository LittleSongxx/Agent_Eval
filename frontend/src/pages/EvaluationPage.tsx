import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  Table,
  Button,
  Card,
  Form,
  Input,
  Select,
  Space,
  message,
  Tag,
  Typography,
  Progress,
  Popconfirm,
  Spin,
  Alert,
  Radio,
  Divider,
  Descriptions,
  Modal,
  Row,
  Col,
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  FileTextOutlined,
  CodeOutlined,
  CloseOutlined,
  CopyOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { EvalTask, Dataset, EndpointTarget, EvalScenario, LLMConfig } from '../types';
import * as api from '../services/api';

const { Title, Text } = Typography;
const { TextArea } = Input;

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

const DEFAULT_DEEPSEEK_ENDPOINT_URL = 'https://api.deepseek.com/v1/chat/completions';
const DEFAULT_DEEPSEEK_AUTHORIZATION = 'Bearer xxx';
const DEFAULT_DEEPSEEK_TEST_INPUT = '请用三句话介绍一下 DeepSeek，并说明它适合做哪些 AI 应用测试。';
const DEFAULT_ENDPOINT_BODY = JSON.stringify(
  {
    model: 'deepseek-chat',
    messages: [
      {
        role: 'user',
        content: '{{user_input}}',
      },
    ],
    temperature: 0.2,
    stream: false,
  },
  null,
  2,
);

const JsonTextArea: React.FC<React.ComponentProps<typeof TextArea>> = ({ className, ...props }) => (
  <TextArea
    {...props}
    className={['json-textarea', className].filter(Boolean).join(' ')}
    spellCheck={false}
  />
);

const FormSectionTitle: React.FC<{ title: string; description?: string }> = ({ title, description }) => (
  <div className="eval-form-section-title">
    <Text strong>{title}</Text>
    {description && <Text type="secondary">{description}</Text>}
  </div>
);

const validateJsonText = (_: unknown, value?: string) => {
  if (!value?.trim()) return Promise.resolve();
  try {
    JSON.parse(value);
    return Promise.resolve();
  } catch {
    return Promise.reject(new Error('请输入合法 JSON'));
  }
};

/** 将 WS 推送的日志尾段合并进已有日志：按公共后缀去重，避免重复行 */
const mergeLogTail = (prev: string, tail: string) => {
  if (!tail) return prev;
  if (!prev) return tail;
  const prevLines = prev.split('\n');
  const tailLines = tail.split('\n');
  let overlap = 0;
  for (let i = 1; i <= Math.min(prevLines.length, tailLines.length); i++) {
    const p = prevLines[prevLines.length - i];
    const t = tailLines[tailLines.length - i];
    if (p !== '' && p === t) overlap = i;
    else break;
  }
  const newLines = tailLines.slice(0, tailLines.length - overlap);
  return newLines.length ? `${prev}\n${newLines.join('\n')}` : prev;
};

const EvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<EvalTask[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [scenarios, setScenarios] = useState<EvalScenario[]>([]);
  const [llmConfigs, setLLMConfigs] = useState<LLMConfig[]>([]);
  const [endpointTargets, setEndpointTargets] = useState<EndpointTarget[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const formCardRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [consoleTaskId, setConsoleTaskId] = useState<number | null>(null);
  const [consoleLogs, setConsoleLogs] = useState('');
  const [consoleStatus, setConsoleStatus] = useState('');
  const [debugging, setDebugging] = useState(false);
  const [debugResult, setDebugResult] = useState<any>(null);
  const [debugModalOpen, setDebugModalOpen] = useState(false);
  const consoleTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const consoleBoxRef = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);
  const [cloneSourceTask, setCloneSourceTask] = useState<EvalTask | null>(null);
  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedScenarioId = Form.useWatch('scenario_id', form);
  const selectedLLMConfigId = Form.useWatch('llm_config_id', form);
  const selectedEvaluationMode = Form.useWatch('evaluation_mode', form) || 'offline';
  const selectedResponseMapping = Form.useWatch('response_mapping', form);
  const [testingEndpoint, setTestingEndpoint] = useState(false);

  const fetchTasks = useCallback(async () => {
    try {
      const data = await api.listEvaluations();
      const normalized = Array.isArray(data) ? data : data?.items || [];
      setTasks(normalized);
      return normalized;
    } catch {
      message.error('加载评测任务失败');
      return [];
    }
  }, []);

  const fetchOptions = useCallback(async () => {
    try {
      const [ds, sc, lc, et] = await Promise.all([
        api.listDatasets(),
        api.listScenarios(),
        api.listLLMConfigs(),
        api.listEndpointTargets(),
      ]);
      setDatasets(Array.isArray(ds) ? ds : []);
      setScenarios(Array.isArray(sc) ? sc : []);
      setLLMConfigs(Array.isArray(lc) ? lc : []);
      setEndpointTargets(Array.isArray(et) ? et : []);
    } catch {
      message.error('加载选项数据失败');
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchTasks(), fetchOptions()]).finally(() =>
      setLoading(false)
    );
  }, [fetchTasks, fetchOptions]);

  useEffect(() => {
    const hasRunning = tasks.some(
      (t) => t.status === 'pending' || t.status === 'running'
    );
    if (hasRunning) {
      timerRef.current = setInterval(fetchTasks, 1000);
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [tasks, fetchTasks]);

  const openConsole = (taskId: number) => {
    setConsoleTaskId(taskId);
    setConsoleLogs('');
    setConsoleStatus('');
    userScrolledUp.current = false;
  };

  const closeConsole = () => {
    setConsoleTaskId(null);
    setConsoleLogs('');
    closeConsoleWs();
  };

  const closeConsoleWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
  }, []);

  // WebSocket 实时进度：任务日志与状态由服务端推送，断开后自动回退轮询
  useEffect(() => {
    if (consoleTaskId === null) return;
    let disposed = false;

    const connectWs = () => {
      const ws = new WebSocket(api.evaluationWsUrl(consoleTaskId));
      wsRef.current = ws;
      ws.onopen = () => { if (!disposed) setWsConnected(true); };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type !== 'task_progress' || disposed) return;
          setConsoleLogs((prev) => mergeLogTail(prev, msg.log_tail || ''));
          setConsoleStatus(msg.status || '');
          setTasks((prev) => prev.map((t) =>
            t.id === msg.task_id
              ? {
                  ...t,
                  status: msg.status,
                  progress: msg.progress,
                  completed_rows: msg.completed_rows,
                  total_rows: msg.total_rows,
                }
              : t
          ));
          if (!userScrolledUp.current && consoleBoxRef.current) {
            setTimeout(() => {
              const box = consoleBoxRef.current;
              if (box) box.scrollTop = box.scrollHeight;
            }, 100);
          }
        } catch { /* 忽略异常消息 */ }
      };
      ws.onclose = () => { if (!disposed) { setWsConnected(false); wsRef.current = null; } };
      ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
    };

    connectWs();

    return () => {
      disposed = true;
      closeConsoleWs();
    };
  }, [consoleTaskId, closeConsoleWs]);

  // 轮询兜底：仅在 WebSocket 未连接时启动
  useEffect(() => {
    if (consoleTaskId === null || wsConnected) return;

    const pollLogs = async () => {
      try {
        const data = await api.getEvaluationLogs(consoleTaskId);
        setConsoleLogs(data.logs || '');
        setConsoleStatus(data.status || '');
        if (!userScrolledUp.current && consoleBoxRef.current) {
          setTimeout(() => {
            const box = consoleBoxRef.current;
            if (box) box.scrollTop = box.scrollHeight;
          }, 100);
        }
      } catch { /* ignore */ }
    };

    pollLogs();
    consoleTimerRef.current = setInterval(pollLogs, 1000);

    return () => {
      if (consoleTimerRef.current) {
        clearInterval(consoleTimerRef.current);
        consoleTimerRef.current = null;
      }
    };
  }, [consoleTaskId, wsConnected]);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      const compatibility = getSelectionCompatibility(values.dataset_id, values.scenario_id);
      if (!compatibility.ok) {
        message.error(compatibility.message);
        return;
      }
      setSubmitting(true);
      const created = await api.createEvaluation(values);
      message.success('评测任务已创建');
      form.resetFields();
      setCloneSourceTask(null);
      setTasks((prev) => [created, ...prev.filter((t) => t.id !== created.id)]);
      fetchTasks();
      openConsole(created.id);
    } catch {
      message.error('创建评测失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDebugEvaluation = async () => {
    try {
      const values = await form.validateFields();
      const compatibility = getSelectionCompatibility(values.dataset_id, values.scenario_id);
      if (!compatibility.ok) {
        message.error(compatibility.message);
        return;
      }
      setDebugging(true);
      const result = await api.debugEvaluation(values);
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

  const handleCancel = async (id: number) => {
    try {
      await api.cancelEvaluation(id);
      message.success('已取消评测');
      fetchTasks();
    } catch {
      message.error('取消失败');
    }
  };

  const handleClone = (task: EvalTask) => {
    const timestamp = new Date().toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
      form.setFieldsValue({
        name: `${task.name} 复跑 ${timestamp}`,
        dataset_id: task.dataset_id,
        scenario_id: task.scenario_id,
        llm_config_id: task.llm_config_id,
        evaluation_mode: task.evaluation_mode || 'offline',
        endpoint_target_id: task.endpoint_target_id,
        target_config: task.target_config,
        response_mapping: task.response_mapping,
        result_save_mode: task.result_save_mode || 'task_only',
        judge_llm_config_ids: task.judge_panel || undefined,
      });
    setCloneSourceTask(task);
    message.success('已复制任务配置，可修改后重新执行');
    setTimeout(() => {
      formCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  };

  const getTaskPercent = (task: EvalTask) => {
    if (task.total_rows && task.total_rows > 0) {
      return Math.min(100, Math.round((task.completed_rows / task.total_rows) * 100));
    }
    return Math.round((task.progress || 0) * 100);
  };

  const getTaskCountText = (task: EvalTask) => {
    const completed = task.completed_rows || 0;
    const total = task.total_rows || 0;
    return total > 0 ? `${completed}/${total}` : `${completed}/-`;
  };

  const getProgressStatus = (status: string): 'active' | 'success' | 'exception' | 'normal' => {
    if (status === 'completed') return 'success';
    if (status === 'failed') return 'exception';
    if (status === 'running') return 'active';
    return 'normal';
  };

  const progressTask =
    (consoleTaskId ? tasks.find((t) => t.id === consoleTaskId) : undefined) ||
    tasks.find((t) => t.status === 'running' || t.status === 'pending');

  const getSelectionCompatibility = (datasetId?: number, scenarioId?: number) => {
    const dataset = datasets.find((d) => d.id === datasetId);
    const scenario = scenarios.find((s) => s.id === scenarioId);
    if (!dataset || !scenario) {
      return { ok: true, message: '' };
    }

    if (dataset.sample_type !== scenario.sample_type) {
      return {
        ok: false,
        message: `样本类型不匹配：数据集是 ${dataset.sample_type}，场景需要 ${scenario.sample_type}`,
      };
    }

    const datasetFields = new Set((dataset.field_schema || []).map((field) => field.name));
    const requiredFields = new Set<string>();
    (scenario.metrics || []).forEach((scenarioMetric) => {
      const metricName = scenarioMetric.metric_definition?.name;
      const fields = metricName ? metricRequiredFields[metricName] || [] : [];
      fields.forEach((field) => requiredFields.add(field));
    });
    const producedFields = getEndpointProducedFields();
    const missingFields = Array.from(requiredFields).filter((field) => {
      if (datasetFields.has(field)) return false;
      return selectedEvaluationMode === 'endpoint' ? !producedFields.has(field) : true;
    });
    if (missingFields.length > 0) {
      return {
        ok: false,
        message: `数据集缺少当前场景必需字段：${missingFields.join(', ')}`,
      };
    }

    return { ok: true, message: '数据集字段满足当前场景的核心指标要求。' };
  };

  const getEndpointProducedFields = () => {
    const mapping = selectedResponseMapping || form.getFieldValue('response_mapping') || {};
    const produced = new Set<string>();
    if (mapping.response_path) produced.add('response');
    if (mapping.retrieved_contexts_path) produced.add('retrieved_contexts');
    if (mapping.tool_calls_path) produced.add('tool_calls');
    if (mapping.retrieved_context_ids_path) produced.add('retrieved_context_ids');
    return produced;
  };

  const handleTestEndpoint = async () => {
    try {
      const values = await form.validateFields([
        ['target_config', 'endpoint_url'],
        ['target_config', 'transport_mode'],
        ['target_config', 'request_body_template'],
      ]);
      setTestingEndpoint(true);
      const rowData = {
        user_input: form.getFieldValue('endpoint_test_user_input') || DEFAULT_DEEPSEEK_TEST_INPUT,
      };
      const result = await api.testEvaluationEndpoint({
        endpoint_target_id: form.getFieldValue('endpoint_target_id'),
        target_config: {
          ...(form.getFieldValue('target_config') || {}),
          request_body_template: form.getFieldValue(['target_config', 'request_body_template']) || DEFAULT_ENDPOINT_BODY,
        },
        response_mapping: form.getFieldValue('response_mapping') || {},
        row_data: rowData,
      });
      if (result.success) {
        Modal.success({
          width: 760,
          title: '接口试跑成功',
          content: (
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <div>
                <Text strong>请求体</Text>
                <pre style={{ maxHeight: 160, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(result.request_body, null, 2)}</pre>
              </div>
              <div>
                <Text strong>解析字段</Text>
                <pre style={{ maxHeight: 160, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(result.extracted_fields || {}, null, 2)}</pre>
              </div>
              {result.mapping_errors && Object.keys(result.mapping_errors).length > 0 && (
                <Alert type="warning" showIcon message="字段映射提示" description={JSON.stringify(result.mapping_errors)} />
              )}
            </Space>
          ),
        });
      } else {
        message.error(`接口试跑失败: ${result.message}`);
      }
    } catch {
      message.error('请先补全接口地址和请求模板');
    } finally {
      setTestingEndpoint(false);
    }
  };

  const applyEndpointTarget = (targetId?: number) => {
    const target = endpointTargets.find((item) => item.id === targetId);
    if (!target) return;
    form.setFieldsValue({
      endpoint_target_id: target.id,
      target_config: {
        endpoint_url: target.endpoint_url,
        transport_mode: target.transport_mode || 'json',
        authorization: '',
        extra_headers: target.extra_headers || '{}',
        request_body_template: target.request_body_template || DEFAULT_ENDPOINT_BODY,
      },
      response_mapping: {
        response_path: target.response_mapping?.response_path || 'choices.0.message.content',
        retrieved_contexts_path: target.response_mapping?.retrieved_contexts_path || '',
        retrieved_context_ids_path: target.response_mapping?.retrieved_context_ids_path || '',
        tool_calls_path: target.response_mapping?.tool_calls_path || '',
      },
      endpoint_test_user_input: target.default_test_input || DEFAULT_DEEPSEEK_TEST_INPUT,
    });
  };

  const selectionCompatibility = getSelectionCompatibility(selectedDatasetId, selectedScenarioId);
  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);
  const customPromptCount = (selectedScenario?.metrics || []).filter((m) => m.prompt_override).length;

  const columns = [
    { title: '任务名称', dataIndex: 'name', key: 'name', width: 180 },
    {
      title: '数据集',
      key: 'dataset',
      width: 140,
      render: (_: unknown, record: EvalTask) =>
        record.dataset?.name ||
        datasets.find((d) => d.id === record.dataset_id)?.name ||
        '-',
    },
    {
      title: '场景',
      key: 'scenario',
      width: 140,
      render: (_: unknown, record: EvalTask) =>
        record.scenario_snapshot?.name ||
        record.scenario?.name ||
        scenarios.find((s) => s.id === record.scenario_id)?.name ||
        '-',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (v: string) => (
        <Tag color={statusColorMap[v] || 'default'}>
          {statusLabelMap[v] || v}
        </Tag>
      ),
    },
    {
      title: '进度',
      key: 'progress',
      width: 200,
      render: (_: unknown, record: EvalTask) => (
        <Space>
          <Progress
            percent={getTaskPercent(record)}
            size="small"
            style={{ width: 120 }}
            status={getProgressStatus(record.status)}
          />
          <span style={{ fontSize: 12, color: '#999' }}>
            {getTaskCountText(record)}
          </span>
        </Space>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (v: string) => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 240,
      render: (_: unknown, record: EvalTask) => (
        <Space>
          <Tag color={(record.evaluation_mode || 'offline') === 'endpoint' ? 'purple' : 'default'}>
            {(record.evaluation_mode || 'offline') === 'endpoint' ? '接口评测' : '已有结果'}
          </Tag>
          <Button
            size="small"
            type="link"
            icon={<CopyOutlined />}
            onClick={() => handleClone(record)}
          >
            克隆
          </Button>
          <Button
            size="small"
            type="link"
            icon={<CodeOutlined />}
            onClick={() => openConsole(record.id)}
          >
            日志
          </Button>
          {record.status === 'completed' && (
            <Button
              size="small"
              type="link"
              icon={<FileTextOutlined />}
              onClick={() => navigate(`/reports/${record.id}`)}
            >
              报告
            </Button>
          )}
          {(record.status === 'running' || record.status === 'pending') && (
            <Popconfirm
              title="确认取消此评测？"
              onConfirm={() => handleCancel(record.id)}
            >
              <Button size="small" danger icon={<StopOutlined />}>
                取消
              </Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

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
    <Spin spinning={loading}>
      <Title level={4}>评测执行</Title>

      <div ref={formCardRef}>
      <Card
        className="eval-create-card"
        title={
          <Space direction="vertical" size={2}>
            <Text strong>{cloneSourceTask ? `克隆评测任务 #${cloneSourceTask.id}` : '新建评测'}</Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              选择数据集、评测场景和 Judge LLM，接口评测可在下方配置请求与字段映射。
            </Text>
          </Space>
        }
        style={{ marginBottom: 24 }}
        extra={cloneSourceTask && (
          <Button
            size="small"
            onClick={() => {
              form.resetFields();
              setCloneSourceTask(null);
            }}
          >
            取消克隆
          </Button>
        )}
      >
        {cloneSourceTask && (
          <Alert
            style={{ marginBottom: 12 }}
            type="info"
            showIcon
            message="已从历史任务复制配置"
            description={`来源任务：${cloneSourceTask.name}。你可以修改任务名称、数据集、场景或 LLM 配置，点击“开始评测”后会生成一条新的历史记录。`}
          />
        )}
        <Form
          form={form}
          layout="vertical"
          className="eval-create-form"
          initialValues={{
            evaluation_mode: 'offline',
            result_save_mode: 'task_only',
            target_config: {
              endpoint_url: DEFAULT_DEEPSEEK_ENDPOINT_URL,
              transport_mode: 'json',
              authorization: DEFAULT_DEEPSEEK_AUTHORIZATION,
              request_body_template: DEFAULT_ENDPOINT_BODY,
              extra_headers: '{}',
            },
            response_mapping: {
              response_path: 'choices.0.message.content',
            },
            endpoint_test_user_input: DEFAULT_DEEPSEEK_TEST_INPUT,
          }}
        >
          <FormSectionTitle title="基础配置" description="这些配置会在任务创建时冻结，后续报告按这次快照展示。" />
          <Row gutter={16}>
            <Col span={6}>
              <Form.Item name="evaluation_mode" label="评测类型">
                <Radio.Group
                  optionType="button"
                  buttonStyle="solid"
                  className="eval-mode-switch"
                  options={[
                    { label: '已有结果评测', value: 'offline' },
                    { label: '接口实时评测', value: 'endpoint' },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item
                name="name"
                label="任务名称"
                rules={[{ required: true, message: '请输入名称' }]}
              >
                <Input placeholder="评测任务名称" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item
                name="dataset_id"
                label="数据集"
                rules={[{ required: true, message: '请选择数据集' }]}
              >
                <Select
                  placeholder="选择数据集"
                  options={datasets.map((d) => ({
                    label: `${d.name} · ${d.sample_type} · ${d.row_count} 条`,
                    value: d.id,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item
                name="scenario_id"
                label="评测场景"
                rules={[{ required: true, message: '请选择场景' }]}
              >
                <Select
                  placeholder="选择评测场景"
                  options={scenarios.map((s) => ({
                    label: `${s.name} · ${s.scene_type}`,
                    value: s.id,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item
                name="llm_config_id"
                label="Judge LLM"
                rules={[{ required: true, message: '请选择 LLM' }]}
              >
                <Select
                  placeholder="选择 LLM 配置"
                  options={llmConfigs.map((c) => ({
                    label: `${c.name} (${c.model_name})`,
                    value: c.id,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item
                name="judge_llm_config_ids"
                label="附加裁判 LLM"
                extra="多裁判独立打分后取均值/多数票，报告展示裁判间一致性（可选）"
              >
                <Select
                  mode="multiple"
                  allowClear
                  placeholder="选择后 LLM 指标由多裁判聚合评分"
                  options={llmConfigs
                    .filter((c) => c.id !== selectedLLMConfigId)
                    .map((c) => ({
                      label: `${c.name} (${c.model_name})`,
                      value: c.id,
                    }))}
                />
              </Form.Item>
            </Col>
          </Row>

          {selectedEvaluationMode === 'endpoint' && (
            <div className="eval-endpoint-panel">
              <FormSectionTitle title="被测接口" description="保存过的接口可直接套用，也可以在这里临时覆盖请求配置。" />
              <Row gutter={16}>
                <Col span={7}>
                  <Form.Item name="endpoint_target_id" label="选择已保存接口">
                    <Select
                      allowClear
                      showSearch
                      optionFilterProp="label"
                      placeholder="选择后自动填充接口配置"
                      onChange={(value) => applyEndpointTarget(value)}
                      options={endpointTargets.map((target) => ({
                        label: `${target.name} · ${target.transport_mode.toUpperCase()}`,
                        value: target.id,
                      }))}
                    />
                  </Form.Item>
                </Col>
                <Col span={9}>
                  <Form.Item
                    name={['target_config', 'endpoint_url']}
                    label="接口地址"
                    rules={[{ required: true, message: '请填写接口地址' }]}
                  >
                    <Input placeholder={DEFAULT_DEEPSEEK_ENDPOINT_URL} />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item name={['target_config', 'transport_mode']} label="返回方式">
                    <Select
                      options={[
                        { label: 'JSON', value: 'json' },
                        { label: 'SSE 流式', value: 'sse' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item name="result_save_mode" label="结果保存">
                    <Select
                      options={[
                        { label: '任务结果', value: 'task_only' },
                        { label: '回写数据集', value: 'write_back' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name={['target_config', 'authorization']} label="Authorization">
                    <Input placeholder={DEFAULT_DEEPSEEK_AUTHORIZATION} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={7}>
                  <Form.Item
                    name={['target_config', 'extra_headers']}
                    label="附加 Headers(JSON)"
                    rules={[{ validator: validateJsonText }]}
                  >
                    <JsonTextArea autoSize={{ minRows: 7, maxRows: 12 }} placeholder={'{\n  "X-Trace-Id": "{{row_id}}"\n}'} />
                  </Form.Item>
                </Col>
                <Col span={10}>
                  <Form.Item
                    name={['target_config', 'request_body_template']}
                    label="请求体模板(JSON)"
                    rules={[
                      { required: true, message: '请填写请求体模板' },
                      { validator: validateJsonText },
                    ]}
                  >
                    <JsonTextArea autoSize={{ minRows: 7, maxRows: 12 }} />
                  </Form.Item>
                </Col>
                <Col span={7}>
                  <Form.Item name="endpoint_test_user_input" label="试跑输入">
                    <TextArea
                      placeholder="输入一条测试问题"
                      autoSize={{ minRows: 7, maxRows: 12 }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider orientation="left" style={{ margin: '4px 0 16px' }}>响应字段映射</Divider>
              <Row gutter={16} align="bottom">
                <Col span={5}>
                  <Form.Item name={['response_mapping', 'response_path']} label="回答字段">
                    <Input placeholder="choices.0.message.content" />
                  </Form.Item>
                </Col>
                <Col span={5}>
                  <Form.Item name={['response_mapping', 'retrieved_contexts_path']} label="上下文字段">
                    <Input placeholder="data.contexts" />
                  </Form.Item>
                </Col>
                <Col span={5}>
                  <Form.Item name={['response_mapping', 'retrieved_context_ids_path']} label="检索ID字段">
                    <Input placeholder="data.context_ids" />
                  </Form.Item>
                </Col>
                <Col span={5}>
                  <Form.Item name={['response_mapping', 'tool_calls_path']} label="工具调用字段">
                    <Input placeholder="trace.tool_calls" />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="连通性">
                    <Button block loading={testingEndpoint} onClick={handleTestEndpoint}>
                      测试接口
                    </Button>
                  </Form.Item>
                </Col>
              </Row>
              <Alert
                style={{ marginBottom: 4 }}
                type="info"
                showIcon
                message="接口实时评测会先逐条调用业务接口，再把映射出的字段交给现有指标评分。"
                description="默认只保存到本次任务结果；选择回写数据集时，会把 response、retrieved_contexts、tool_calls 等映射结果补充回原始数据行。"
              />
            </div>
          )}

          <div className="eval-create-actions">
            <Space>
              <Button
                icon={<CodeOutlined />}
                loading={debugging}
                disabled={!selectionCompatibility.ok}
                onClick={handleDebugEvaluation}
              >
                评测流程验证
              </Button>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                loading={submitting}
                disabled={!selectionCompatibility.ok}
                onClick={handleCreate}
              >
                开始评测
              </Button>
            </Space>
          </div>
        </Form>
        {selectedDatasetId && selectedScenarioId && (
          <Alert
            style={{ marginTop: 12 }}
            type={selectionCompatibility.ok ? 'success' : 'warning'}
            showIcon
            message={selectionCompatibility.ok ? '数据集与场景字段匹配' : '数据集与场景不匹配'}
            description={
              <span>
                {selectionCompatibility.message}
                {selectionCompatibility.ok && customPromptCount > 0 && (
                  <span> 当前场景包含 {customPromptCount} 个业务自定义评测提示词，新建任务会冻结这些评测口径。</span>
                )}
              </span>
            }
          />
        )}
      </Card>
      </div>

      {progressTask && (
        <Card
          title="当前评测进度"
          style={{ marginBottom: 24, borderColor: '#d6e4ff' }}
          extra={
            <Tag color={statusColorMap[progressTask.status] || 'default'}>
              {statusLabelMap[progressTask.status] || progressTask.status}
            </Tag>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }} size={12}>
            <Space wrap>
              <Text strong>{progressTask.name}</Text>
              <Text type="secondary">
                总计 {progressTask.total_rows || 0} 条，已完成 {progressTask.completed_rows || 0} 条
              </Text>
              {progressTask.status === 'running' && (
                <Text type="secondary">{wsConnected ? 'WebSocket 实时推送中' : '每 1 秒自动刷新一次（WS 断开降级）'}</Text>
              )}
            </Space>
            <Progress
              percent={getTaskPercent(progressTask)}
              status={getProgressStatus(progressTask.status)}
              strokeWidth={12}
            />
            {progressTask.error_message && (
              <Text type="danger">{progressTask.error_message}</Text>
            )}
          </Space>
        </Card>
      )}

      <Card title="评测历史记录" style={{ marginBottom: 24 }}>
        <Table
          rowKey="id"
          columns={columns}
          dataSource={tasks}
          pagination={{
            pageSize: 10,
            showTotal: (t) => `共 ${t} 条`,
          }}
        />
      </Card>

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
                  <Text type="secondary">
                    样本行 #{debugResult.dataset_row?.row_index ?? '-'}
                  </Text>
                </Space>
              }
            />
            {debugResult.errors?.length > 0 && (
              <Alert
                type="error"
                showIcon
                message="错误"
                description={debugResult.errors.join('；')}
              />
            )}
            {debugResult.warnings?.length > 0 && (
              <Alert
                type="warning"
                showIcon
                message="风险提示"
                description={debugResult.warnings.join('；')}
              />
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
                  {debugResult.endpoint_trace.mapping_errors && Object.keys(debugResult.endpoint_trace.mapping_errors).length > 0 && (
                    <Alert type="warning" showIcon message="字段映射提示" description={JSON.stringify(debugResult.endpoint_trace.mapping_errors)} />
                  )}
                  <Text strong>原始响应</Text>
                  {renderJsonBlock(debugResult.endpoint_trace.raw_response || debugResult.endpoint_trace.error)}
                </Space>
              </Card>
            )}

            <Card size="small" title="3. Judge Prompt 与返回结果">
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                {(debugResult.judge_traces || []).map((trace: any) => (
                  <Card
                    key={trace.metric_name}
                    size="small"
                    title={
                      <Space wrap>
                        <Text strong>{trace.metric_display_name || trace.metric_name}</Text>
                        <Tag>{trace.metric_type}</Tag>
                        {trace.parsed_result?.score !== undefined && (
                          <Tag color={trace.parsed_result?.score == null ? 'red' : 'blue'}>
                            score: {String(trace.parsed_result?.score)}
                          </Tag>
                        )}
                      </Space>
                    }
                  >
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

      {consoleTaskId !== null && (
        <div style={{ marginTop: 16 }}>
          <div
            style={{
              background: '#1e1e1e',
              borderRadius: '8px 8px 0 0',
              padding: '8px 16px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Space>
              <CodeOutlined style={{ color: '#4ec9b0' }} />
              <Text style={{ color: '#ccc', fontSize: 13 }}>
                评测日志 — 任务 #{consoleTaskId}
              </Text>
              <Tag
                color={
                  consoleStatus === 'running' ? 'orange' :
                  consoleStatus === 'completed' ? 'green' :
                  consoleStatus === 'failed' ? 'red' : 'blue'
                }
                style={{ fontSize: 11 }}
              >
                {statusLabelMap[consoleStatus] || consoleStatus}
              </Tag>
              {wsConnected && (
                <Tag color="cyan" style={{ fontSize: 11 }}>实时推送</Tag>
              )}
            </Space>
            <Button
              type="text"
              size="small"
              icon={<CloseOutlined style={{ color: '#999' }} />}
              onClick={closeConsole}
            />
          </div>
          <div
            ref={consoleBoxRef}
            onScroll={() => {
              const el = consoleBoxRef.current;
              if (!el) return;
              const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
              userScrolledUp.current = !atBottom;
            }}
            style={{
              background: '#1e1e1e',
              borderRadius: '0 0 8px 8px',
              padding: '12px 16px',
              height: 360,
              overflowY: 'auto',
              fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
              fontSize: 13,
              lineHeight: 1.7,
              color: '#d4d4d4',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {consoleLogs ? consoleLogs.split('\n').map((line, i) => {
              let color = '#d4d4d4';
              if (line.includes('✓')) color = '#4ec9b0';
              else if (line.includes('✗')) color = '#f44747';
              else if (line.includes('⚠')) color = '#dcdcaa';
              else if (line.includes('══')) color = '#569cd6';
              else if (line.includes('──')) color = '#808080';
              else if (line.includes('▸')) color = '#9cdcfe';
              return (
                <div key={i} style={{ color, minHeight: 20 }}>
                  {line}
                </div>
              );
            }) : (
              <div style={{ color: '#808080' }}>等待日志输出...</div>
            )}
            <div ref={consoleEndRef} />
          </div>
        </div>
      )}
    </Spin>
  );
};

export default EvaluationPage;

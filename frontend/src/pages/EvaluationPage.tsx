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
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  FileTextOutlined,
  CodeOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { EvalTask, Dataset, EvalScenario, LLMConfig } from '../types';
import * as api from '../services/api';

const { Title, Text } = Typography;

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

const EvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<EvalTask[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [scenarios, setScenarios] = useState<EvalScenario[]>([]);
  const [llmConfigs, setLLMConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [consoleTaskId, setConsoleTaskId] = useState<number | null>(null);
  const [consoleLogs, setConsoleLogs] = useState('');
  const [consoleStatus, setConsoleStatus] = useState('');
  const consoleTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const consoleBoxRef = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);
  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedScenarioId = Form.useWatch('scenario_id', form);

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
      const [ds, sc, lc] = await Promise.all([
        api.listDatasets(),
        api.listScenarios(),
        api.listLLMConfigs(),
      ]);
      setDatasets(Array.isArray(ds) ? ds : []);
      setScenarios(Array.isArray(sc) ? sc : []);
      setLLMConfigs(Array.isArray(lc) ? lc : []);
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
    if (consoleTimerRef.current) {
      clearInterval(consoleTimerRef.current);
      consoleTimerRef.current = null;
    }
  };

  useEffect(() => {
    if (consoleTaskId === null) return;

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
  }, [consoleTaskId]);

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
      setTasks((prev) => [created, ...prev.filter((t) => t.id !== created.id)]);
      fetchTasks();
      openConsole(created.id);
    } catch {
      message.error('创建评测失败');
    } finally {
      setSubmitting(false);
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
    const missingFields = Array.from(requiredFields).filter((field) => !datasetFields.has(field));
    if (missingFields.length > 0) {
      return {
        ok: false,
        message: `数据集缺少当前场景必需字段：${missingFields.join(', ')}`,
      };
    }

    return { ok: true, message: '数据集字段满足当前场景的核心指标要求。' };
  };

  const selectionCompatibility = getSelectionCompatibility(selectedDatasetId, selectedScenarioId);

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
      width: 180,
      render: (_: unknown, record: EvalTask) => (
        <Space>
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

  return (
    <Spin spinning={loading}>
      <Title level={4}>评测执行</Title>

      <Card title="新建评测" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="inline"
          style={{ flexWrap: 'wrap', gap: 8 }}
        >
          <Form.Item
            name="name"
            rules={[{ required: true, message: '请输入名称' }]}
          >
            <Input placeholder="评测任务名称" style={{ width: 200 }} />
          </Form.Item>
          <Form.Item
            name="dataset_id"
            rules={[{ required: true, message: '请选择数据集' }]}
          >
            <Select
              placeholder="选择数据集"
              style={{ width: 200 }}
              options={datasets.map((d) => ({
                label: `${d.name} · ${d.sample_type} · ${d.row_count} 条`,
                value: d.id,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="scenario_id"
            rules={[{ required: true, message: '请选择场景' }]}
          >
            <Select
              placeholder="选择评测场景"
              style={{ width: 200 }}
              options={scenarios.map((s) => ({
                label: `${s.name} · ${s.scene_type}`,
                value: s.id,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="llm_config_id"
            rules={[{ required: true, message: '请选择 LLM' }]}
          >
            <Select
              placeholder="选择 LLM 配置"
              style={{ width: 200 }}
              options={llmConfigs.map((c) => ({
                label: `${c.name} (${c.model_name})`,
                value: c.id,
              }))}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              loading={submitting}
              disabled={!selectionCompatibility.ok}
              onClick={handleCreate}
            >
              开始评测
            </Button>
          </Form.Item>
        </Form>
        {selectedDatasetId && selectedScenarioId && (
          <Alert
            style={{ marginTop: 12 }}
            type={selectionCompatibility.ok ? 'success' : 'warning'}
            showIcon
            message={selectionCompatibility.ok ? '数据集与场景字段匹配' : '数据集与场景不匹配'}
            description={selectionCompatibility.message}
          />
        )}
      </Card>

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
                <Text type="secondary">每 1 秒自动刷新一次</Text>
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

      <Table
        rowKey="id"
        columns={columns}
        dataSource={tasks}
        pagination={{
          pageSize: 10,
          showTotal: (t) => `共 ${t} 条`,
        }}
      />

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

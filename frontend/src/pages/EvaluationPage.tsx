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
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { EvalTask, Dataset, EvalScenario, LLMConfig } from '../types';
import * as api from '../services/api';

const { Title } = Typography;

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

  const fetchTasks = useCallback(async () => {
    try {
      const data = await api.listEvaluations();
      setTasks(Array.isArray(data) ? data : data?.items || []);
    } catch {
      message.error('加载评测任务失败');
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
      timerRef.current = setInterval(fetchTasks, 3000);
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [tasks, fetchTasks]);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      await api.createEvaluation(values);
      message.success('评测任务已创建');
      form.resetFields();
      fetchTasks();
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
            percent={Math.round(record.progress * 100)}
            size="small"
            style={{ width: 120 }}
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <span style={{ fontSize: 12, color: '#999' }}>
            {record.completed_rows}/{record.total_rows}
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
          {record.status === 'completed' && (
            <Button
              size="small"
              type="link"
              icon={<FileTextOutlined />}
              onClick={() => navigate(`/reports/${record.id}`)}
            >
              查看报告
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
              options={datasets.map((d) => ({ label: d.name, value: d.id }))}
            />
          </Form.Item>
          <Form.Item
            name="scenario_id"
            rules={[{ required: true, message: '请选择场景' }]}
          >
            <Select
              placeholder="选择评测场景"
              style={{ width: 200 }}
              options={scenarios.map((s) => ({ label: s.name, value: s.id }))}
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
              onClick={handleCreate}
            >
              开始评测
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Table
        rowKey="id"
        columns={columns}
        dataSource={tasks}
        pagination={{
          pageSize: 10,
          showTotal: (t) => `共 ${t} 条`,
        }}
      />
    </Spin>
  );
};

export default EvaluationPage;

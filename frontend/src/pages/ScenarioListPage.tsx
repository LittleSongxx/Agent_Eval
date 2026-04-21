import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Space,
  message,
  Popconfirm,
  Tag,
  Typography,
  Card,
  Row,
  Col,
  Badge,
  Spin,
  InputNumber,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  RobotOutlined,
  NodeIndexOutlined,
  MessageOutlined,
} from '@ant-design/icons';
import type { EvalScenario, MetricDefinition } from '../types';
import * as api from '../services/api';

const { Title, Paragraph } = Typography;

interface SelectedMetric {
  metric_definition_id: number;
  weight: number;
  pass_threshold: number | null;
}

const presetCards = [
  {
    key: 'rag',
    title: 'RAG 评测',
    icon: <NodeIndexOutlined style={{ fontSize: 32, color: '#1677ff' }} />,
    description: '评测检索增强生成的准确性、忠实度和上下文相关性',
    scene_type: 'rag',
  },
  {
    key: 'agent',
    title: 'Agent 评测',
    icon: <RobotOutlined style={{ fontSize: 32, color: '#52c41a' }} />,
    description: '评测 Agent 工具调用能力、任务完成度和推理正确性',
    scene_type: 'agent',
  },
  {
    key: 'multi_turn',
    title: '多轮对话评测',
    icon: <MessageOutlined style={{ fontSize: 32, color: '#722ed1' }} />,
    description: '评测多轮对话中的连贯性、上下文理解和回复质量',
    scene_type: 'multi_turn',
  },
];

const sceneTypeOptions = [
  { label: 'RAG', value: 'rag' },
  { label: 'Agent', value: 'agent' },
  { label: '多轮对话', value: 'multi_turn' },
  { label: '通用', value: 'general' },
];

const sampleTypeByScene: Record<string, string[]> = {
  rag: ['single_turn'],
  agent: ['single_turn', 'multi_turn'],
  multi_turn: ['multi_turn'],
  general: ['single_turn', 'multi_turn'],
};

const ScenarioListPage: React.FC = () => {
  const [scenarios, setScenarios] = useState<EvalScenario[]>([]);
  const [presetScenarios, setPresetScenarios] = useState<EvalScenario[]>([]);
  const [metrics, setMetrics] = useState<MetricDefinition[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const [selectedSceneType, setSelectedSceneType] = useState<string>('rag');
  const [selectedMetrics, setSelectedMetrics] = useState<SelectedMetric[]>([]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [scenarioData, metricData] = await Promise.all([
        api.listScenarios(),
        api.listMetrics(),
      ]);
      const allScenarios: EvalScenario[] = Array.isArray(scenarioData) ? scenarioData : [];
      setPresetScenarios(allScenarios.filter((s) => s.is_preset));
      setScenarios(allScenarios.filter((s) => !s.is_preset));
      setMetrics(Array.isArray(metricData) ? metricData : []);
    } catch {
      message.error('加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      if (selectedMetrics.length === 0) {
        message.error('请至少选择一个评测指标');
        return;
      }
      setSubmitting(true);
      await api.createScenario({
        name: values.name,
        description: values.description || '',
        scene_type: values.scene_type,
        sample_type: values.sample_type,
        metrics: selectedMetrics,
      });
      message.success('场景创建成功');
      setModalOpen(false);
      form.resetFields();
      setSelectedMetrics([]);
      fetchData();
    } catch {
      message.error('创建失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.deleteScenario(id);
      message.success('删除成功');
      fetchData();
    } catch {
      message.error('删除失败');
    }
  };

  const filteredMetrics = metrics.filter(
    (m) => m.category === selectedSceneType || m.category === 'custom' || m.category === 'general'
  );

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    {
      title: '场景类型',
      dataIndex: 'scene_type',
      key: 'scene_type',
      width: 120,
      render: (v: string) => <Tag color="blue">{v}</Tag>,
    },
    {
      title: '样本类型',
      dataIndex: 'sample_type',
      key: 'sample_type',
      width: 120,
      render: (v: string) => <Tag color="purple">{v}</Tag>,
    },
    {
      title: '指标数',
      dataIndex: 'metrics',
      key: 'metric_count',
      width: 80,
      render: (m: any[]) => m?.length || 0,
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
      width: 100,
      render: (_: unknown, record: EvalScenario) => (
        <Popconfirm title="确认删除此场景？" onConfirm={() => handleDelete(record.id)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <Spin spinning={loading}>
      <Title level={4}>场景管理</Title>

      <Title level={5} style={{ marginTop: 24 }}>预置模板</Title>
      <Row gutter={16} style={{ marginBottom: 32 }}>
        {presetCards.map((preset) => {
          const matched = presetScenarios.find((s) => s.scene_type === preset.scene_type);
          return (
            <Col span={8} key={preset.key}>
              <Card hoverable style={{ textAlign: 'center', height: '100%' }}>
                <div style={{ marginBottom: 12 }}>{preset.icon}</div>
                <Title level={5} style={{ margin: 0 }}>{preset.title}</Title>
                <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 12 }}>
                  {preset.description}
                </Paragraph>
                {matched && (
                  <Badge
                    count={`${matched.metrics?.length || 0} 个指标`}
                    style={{ backgroundColor: '#1677ff' }}
                  />
                )}
              </Card>
            </Col>
          );
        })}
      </Row>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={5} style={{ margin: 0 }}>自定义场景</Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
          新建场景
        </Button>
      </div>

      <Table rowKey="id" columns={columns} dataSource={scenarios} pagination={false} />

      <Modal
        title="新建评测场景"
        open={modalOpen}
        onOk={handleCreate}
        confirmLoading={submitting}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
          setSelectedMetrics([]);
        }}
        okText="创建"
        cancelText="取消"
        width={720}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{ scene_type: 'rag', sample_type: 'single_turn' }}
          style={{ marginTop: 16 }}
        >
          <Form.Item
            name="name"
            label="场景名称"
            rules={[{ required: true, message: '请输入场景名称' }]}
          >
            <Input placeholder="例如: 客服问答 RAG 评测" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} placeholder="场景描述" />
          </Form.Item>
          <Form.Item name="scene_type" label="场景类型" rules={[{ required: true }]}>
            <Select
              options={sceneTypeOptions}
              onChange={(val: string) => {
                setSelectedSceneType(val);
                const types = sampleTypeByScene[val] || ['single_turn'];
                form.setFieldValue('sample_type', types[0]);
                setSelectedMetrics([]);
              }}
            />
          </Form.Item>
          <Form.Item name="sample_type" label="样本类型" rules={[{ required: true }]}>
            <Select
              options={(sampleTypeByScene[selectedSceneType] || ['single_turn']).map((t) => ({
                label: t === 'single_turn' ? '单轮对话' : '多轮对话',
                value: t,
              }))}
            />
          </Form.Item>
        </Form>

        <div style={{ marginTop: 8 }}>
          <Typography.Text strong>评测指标</Typography.Text>
          <Select
            mode="multiple"
            placeholder="选择评测指标"
            style={{ width: '100%', marginTop: 8, marginBottom: 12 }}
            options={filteredMetrics.map((m) => ({
              label: `${m.display_name} (${m.category})`,
              value: m.id,
            }))}
            onChange={(ids: number[]) => {
              setSelectedMetrics(
                ids.map((mid) => {
                  const existing = selectedMetrics.find((s) => s.metric_definition_id === mid);
                  return existing ?? { metric_definition_id: mid, weight: 1.0, pass_threshold: null };
                })
              );
            }}
            value={selectedMetrics.map((m) => m.metric_definition_id)}
          />
          {selectedMetrics.length > 0 && (
            <Card size="small" title="指标配置">
              {selectedMetrics.map((sm) => {
                const def = metrics.find((m) => m.id === sm.metric_definition_id);
                return (
                  <div key={sm.metric_definition_id} style={{ display: 'flex', alignItems: 'center', marginBottom: 8, gap: 12 }}>
                    <span style={{ width: 140 }}>{def?.display_name ?? sm.metric_definition_id}</span>
                    <span>权重:</span>
                    <InputNumber
                      min={0} max={10} step={0.1}
                      value={sm.weight}
                      onChange={(v) => setSelectedMetrics((prev) =>
                        prev.map((m) => m.metric_definition_id === sm.metric_definition_id ? { ...m, weight: v ?? 1 } : m)
                      )}
                      style={{ width: 80 }}
                    />
                    <span>通过阈值:</span>
                    <InputNumber
                      min={0} max={1} step={0.05}
                      value={sm.pass_threshold ?? undefined}
                      onChange={(v) => setSelectedMetrics((prev) =>
                        prev.map((m) => m.metric_definition_id === sm.metric_definition_id ? { ...m, pass_threshold: v } : m)
                      )}
                      placeholder="可选"
                      style={{ width: 80 }}
                    />
                  </div>
                );
              })}
            </Card>
          )}
        </div>
      </Modal>
    </Spin>
  );
};

export default ScenarioListPage;

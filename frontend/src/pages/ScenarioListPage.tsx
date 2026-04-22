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
import { MetricHelpButton, MetricHelpIcon, MetricHelpDrawer } from '../components/MetricHelpDrawer';
import { getMetricInfo, getMetricLayer, getMetricLayerIndex, groupMetricNames, isMetricAvailableForScene } from '../utils/metricLayers';

const { Title, Paragraph, Text } = Typography;

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
  const [helpMetric, setHelpMetric] = useState<string | null>(null);
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

  const filteredMetrics = metrics
    .filter((m) => isMetricAvailableForScene(m, selectedSceneType))
    .sort((a, b) => {
      const layerDiff = getMetricLayerIndex(a.name) - getMetricLayerIndex(b.name);
      if (layerDiff !== 0) {
        return layerDiff;
      }
      return a.display_name.localeCompare(b.display_name, 'zh-CN');
    });

  const getMetricDisplayName = (metricDefId: number) => {
    const def = metrics.find((m) => m.id === metricDefId);
    return def?.display_name || `指标#${metricDefId}`;
  };

  const getScenarioMetricName = (scenarioMetric: any) =>
    scenarioMetric.metric_definition?.name || `metric_${scenarioMetric.metric_definition_id}`;

  const getScenarioMetricLabel = (scenarioMetric: any) => {
    const metricName = getScenarioMetricName(scenarioMetric);
    const info = getMetricInfo(metricName);
    if (info.displayName !== metricName) {
      return info.shortName;
    }
    return scenarioMetric.metric_definition?.display_name || getMetricDisplayName(scenarioMetric.metric_definition_id);
  };

  const renderScenarioMetricGroups = (metricList: any[], compact = false) => {
    if (!metricList || metricList.length === 0) return <Tag>无指标</Tag>;
    const metricByName = new Map(
      metricList.map((scenarioMetric: any) => [getScenarioMetricName(scenarioMetric), scenarioMetric])
    );

    return (
      <Space direction="vertical" size={4} style={{ width: '100%' }}>
        {groupMetricNames(metricList.map(getScenarioMetricName)).map((group) => (
          <div key={group.key}>
            <Tag color={group.color} style={{ marginRight: 8 }}>{group.name}</Tag>
            <Space size={[4, 4]} wrap>
              {group.metrics.map((metricName) => {
                const scenarioMetric: any = metricByName.get(metricName);
                return (
                  <Tag key={scenarioMetric?.id || metricName} color="cyan" style={{ marginRight: 0 }}>
                    {getScenarioMetricLabel(scenarioMetric)}
                    {!compact && scenarioMetric?.pass_threshold != null && (
                      <span style={{ color: '#999', marginLeft: 4 }}>≥{scenarioMetric.pass_threshold}</span>
                    )}
                    <MetricHelpIcon metricName={metricName} onClick={() => setHelpMetric(metricName)} />
                  </Tag>
                );
              })}
            </Space>
          </div>
        ))}
      </Space>
    );
  };

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name', width: 180 },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    {
      title: '场景类型',
      dataIndex: 'scene_type',
      key: 'scene_type',
      width: 100,
      render: (v: string) => <Tag color="blue">{v}</Tag>,
    },
    {
      title: '样本类型',
      dataIndex: 'sample_type',
      key: 'sample_type',
      width: 100,
      render: (v: string) => <Tag color="purple">{v === 'single_turn' ? '单轮' : '多轮'}</Tag>,
    },
    {
      title: '评测指标',
      dataIndex: 'metrics',
      key: 'metrics_display',
      render: (metricList: any[]) => renderScenarioMetricGroups(metricList),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (v: string) => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={4} style={{ margin: 0 }}>场景管理</Title>
        <MetricHelpButton />
      </div>
      <MetricHelpDrawer open={!!helpMetric} metricName={helpMetric} onClose={() => setHelpMetric(null)} />

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
                  <Space direction="vertical" size={8}>
                    <Badge
                      count={`${matched.metrics?.length || 0} 个指标`}
                      style={{ backgroundColor: '#1677ff' }}
                    />
                    {renderScenarioMetricGroups(matched.metrics || [], true)}
                  </Space>
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
            optionLabelProp="label"
            options={filteredMetrics.map((m) => {
              const desc = m.config?.description || '';
              const layer = getMetricLayer(m.name);
              const info = getMetricInfo(m.name);
              return {
                label: m.display_name,
                value: m.id,
                desc,
                layer,
                info,
              };
            })}
            optionRender={(option) => (
              <div>
                <div style={{ fontWeight: 500 }}>
                  <Tag color={option.data.layer.color} style={{ marginRight: 6 }}>
                    {option.data.layer.name}
                  </Tag>
                  {option.label}
                  {option.data.info.defaultEnabled === false && (
                    <Text type="secondary" style={{ marginLeft: 6, fontSize: 12 }}>默认不启用</Text>
                  )}
                </div>
                {option.data.desc && (
                  <div style={{ fontSize: 12, color: '#888', lineHeight: 1.4 }}>
                    {option.data.desc}
                  </div>
                )}
              </div>
            )}
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
              <Space direction="vertical" style={{ width: '100%' }} size={12}>
                {groupMetricNames(selectedMetrics.map((sm) => metrics.find((m) => m.id === sm.metric_definition_id)?.name || '').filter(Boolean)).map((group) => (
                  <div key={group.key}>
                    <div style={{ marginBottom: 8 }}>
                      <Tag color={group.color}>{group.name}</Tag>
                      <Text type="secondary" style={{ fontSize: 12 }}>{group.description}</Text>
                    </div>
                    {group.metrics.map((metricName) => {
                      const def = metrics.find((m) => m.name === metricName);
                      const sm = selectedMetrics.find((m) => m.metric_definition_id === def?.id);
                      if (!def || !sm) return null;
                      return (
                        <div key={sm.metric_definition_id} style={{ display: 'flex', alignItems: 'center', marginBottom: 8, gap: 12 }}>
                          <span style={{ width: 220, flexShrink: 0 }}>
                            {def.display_name}
                            <MetricHelpIcon metricName={def.name} onClick={() => setHelpMetric(def.name)} />
                          </span>
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
                  </div>
                ))}
              </Space>
            </Card>
          )}
        </div>
      </Modal>
    </Spin>
  );
};

export default ScenarioListPage;

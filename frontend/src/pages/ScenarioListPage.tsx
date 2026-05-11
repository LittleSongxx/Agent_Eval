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
  Alert,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
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
  prompt_override?: string | null;
}

interface ScenarioMetricEntry {
  key: string;
  metricName: string;
  label: string;
  scenarioMetric: any;
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
  const [editingScenario, setEditingScenario] = useState<EvalScenario | null>(null);
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

  const resetModalState = () => {
    setModalOpen(false);
    setEditingScenario(null);
    form.resetFields();
    setSelectedSceneType('rag');
    setSelectedMetrics([]);
  };

  const openCreateModal = () => {
    setEditingScenario(null);
    form.resetFields();
    form.setFieldsValue({ scene_type: 'rag', sample_type: 'single_turn' });
    setSelectedSceneType('rag');
    setSelectedMetrics([]);
    setModalOpen(true);
  };

  const openEditModal = (scenario: EvalScenario) => {
    setEditingScenario(scenario);
    setSelectedSceneType(scenario.scene_type);
    form.setFieldsValue({
      name: scenario.name,
      description: scenario.description || '',
      scene_type: scenario.scene_type,
      sample_type: scenario.sample_type,
    });
    setSelectedMetrics(
      (scenario.metrics || []).map((scenarioMetric) => ({
        metric_definition_id: scenarioMetric.metric_definition_id,
        weight: scenarioMetric.weight ?? 1.0,
        pass_threshold: scenarioMetric.pass_threshold ?? null,
        prompt_override: scenarioMetric.prompt_override || null,
      }))
    );
    setModalOpen(true);
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      if (selectedMetrics.length === 0) {
        message.error('请至少选择一个评测指标');
        return;
      }
      setSubmitting(true);
      const payload = {
        name: values.name,
        description: values.description || '',
        scene_type: values.scene_type,
        sample_type: values.sample_type,
        metrics: selectedMetrics,
      };
      if (editingScenario) {
        await api.updateScenario(editingScenario.id, payload);
        message.success('场景已更新，新建任务会使用最新配置');
      } else {
        await api.createScenario(payload);
        message.success('场景创建成功');
      }
      resetModalState();
      fetchData();
    } catch (error: any) {
      if (!error?.errorFields) {
        message.error(editingScenario ? '更新失败' : '创建失败');
      }
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

  const getScenarioMetricEntries = (metricList: any[] = []): ScenarioMetricEntry[] =>
    metricList.map((scenarioMetric: any, index: number) => {
      const metricName = getScenarioMetricName(scenarioMetric);
      return {
        key: `${scenarioMetric?.id || scenarioMetric?.metric_definition_id || metricName}-${index}`,
        metricName,
        label: getScenarioMetricLabel(scenarioMetric),
        scenarioMetric,
      };
    });

  const getScenarioMetricCount = (metricList: any[] = []) => getScenarioMetricEntries(metricList).length;

  const getScenarioMetricGroups = (metricList: any[] = []) => {
    const entries = getScenarioMetricEntries(metricList);
    const layerOrder = Array.from(new Set(entries.map((entry) => getMetricLayer(entry.metricName).key)))
      .sort((a, b) => {
        const aMetric = entries.find((entry) => getMetricLayer(entry.metricName).key === a)?.metricName || '';
        const bMetric = entries.find((entry) => getMetricLayer(entry.metricName).key === b)?.metricName || '';
        return getMetricLayerIndex(aMetric) - getMetricLayerIndex(bMetric);
      });

    return layerOrder.map((layerKey) => {
      const layerEntries = entries.filter((entry) => getMetricLayer(entry.metricName).key === layerKey);
      return {
        ...getMetricLayer(layerEntries[0].metricName),
        entries: layerEntries,
      };
    });
  };

  const renderScenarioMetricGroups = (metricList: any[], compact = false) => {
    if (!metricList || metricList.length === 0) return <Tag>无指标</Tag>;
    const groups = getScenarioMetricGroups(metricList);

    return (
      <Space direction="vertical" size={compact ? 8 : 10} style={{ width: '100%' }}>
        {groups.map((group) => (
          <div key={group.key} style={{ textAlign: 'left' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 8,
                marginBottom: 6,
                padding: compact ? '3px 8px' : '5px 10px',
                borderLeft: `3px solid ${group.headerBorder}`,
                background: group.headerBg,
                color: group.headerText,
                borderRadius: 4,
                fontSize: compact ? 12 : 13,
                fontWeight: 500,
              }}
            >
              <span>{group.name}</span>
              <span>{group.entries.length} 项</span>
            </div>
            <Space size={[4, 4]} wrap style={{ width: '100%' }}>
              {group.entries.map((entry) => {
                const scenarioMetric = entry.scenarioMetric;
                return (
                  <Tag key={entry.key} color={group.color} style={{ marginRight: 0 }}>
                    {entry.label}
                    {!compact && scenarioMetric?.pass_threshold != null && (
                      <span style={{ color: '#999', marginLeft: 4 }}>≥{scenarioMetric.pass_threshold}</span>
                    )}
                    {!compact && scenarioMetric?.prompt_override && (
                      <span style={{ color: '#1677ff', marginLeft: 4 }}>场景覆盖</span>
                    )}
                    <MetricHelpIcon metricName={entry.metricName} onClick={() => setHelpMetric(entry.metricName)} />
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
      render: (metricList: any[]) => (
        <Space direction="vertical" size={6} style={{ width: '100%' }}>
          <Badge
            count={`${getScenarioMetricCount(metricList || [])} 个指标`}
            style={{ backgroundColor: '#1677ff' }}
          />
          {renderScenarioMetricGroups(metricList)}
        </Space>
      ),
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
      width: 150,
      render: (_: unknown, record: EvalScenario) => (
        <Space size={8}>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEditModal(record)}>
            编辑
          </Button>
          <Popconfirm title="确认删除此场景？" onConfirm={() => handleDelete(record.id)}>
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
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
                      count={`${getScenarioMetricCount(matched.metrics || [])} 个指标`}
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
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
          新建场景
        </Button>
      </div>

      <Table rowKey="id" columns={columns} dataSource={scenarios} pagination={false} />

      <Modal
        title={editingScenario ? '编辑评测场景' : '新建评测场景'}
        open={modalOpen}
        onOk={handleSave}
        confirmLoading={submitting}
        onCancel={resetModalState}
        okText={editingScenario ? '保存' : '创建'}
        cancelText="取消"
        width={720}
      >
        {editingScenario && (
          <Paragraph type="secondary" style={{ marginTop: 0, marginBottom: 12 }}>
            修改只影响之后新建的评测任务；已有任务会继续使用创建任务时保存的场景快照。
          </Paragraph>
        )}
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
            showSearch
            placeholder="选择评测指标"
            style={{ width: '100%', marginTop: 8, marginBottom: 12 }}
            optionLabelProp="label"
            optionFilterProp="searchText"
            filterOption={(input, option) =>
              String(option?.searchText || '').includes(input.trim().toLowerCase())
            }
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
                searchText: [
                  m.display_name,
                  m.name,
                  m.category,
                  m.metric_type,
                  desc,
                  m.is_builtin ? '内置 builtin' : '自定义 custom',
                ].filter(Boolean).join(' ').toLowerCase(),
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
                  return existing ?? { metric_definition_id: mid, weight: 1.0, pass_threshold: null, prompt_override: null };
                })
              );
            }}
            value={selectedMetrics.map((m) => m.metric_definition_id)}
          />
          {selectedMetrics.length > 0 && (
            <Card size="small" title="指标配置">
              <Alert
                type="info"
                showIcon
                style={{ marginBottom: 12 }}
                message="这里填写场景覆盖规则"
                description="指标管理里维护默认评分规则；这里留空时使用指标默认规则，填写后只在当前场景覆盖该指标规则。适合写当前业务线、接口版本或测试目标的特殊扣分/通过标准。系统仍会统一要求评测模型返回 score 和 reason。"
              />
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
                        <div
                          key={sm.metric_definition_id}
                          style={{
                            marginBottom: 12,
                            padding: 12,
                            border: '1px solid #f0f0f0',
                            borderRadius: 6,
                            background: '#fff',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8, gap: 12, flexWrap: 'wrap' }}>
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
                            {sm.prompt_override && <Tag color="processing">已覆盖默认规则</Tag>}
                          </div>
                          <Input.TextArea
                            rows={3}
                            allowClear
                            placeholder="可选：填写该指标在当前场景下的覆盖规则。例如当前业务必须覆盖哪些规则、哪些情况必须扣分、低于多少分算失败。留空则使用指标管理中的默认评分规则。"
                            value={sm.prompt_override || ''}
                            onChange={(e) => setSelectedMetrics((prev) =>
                              prev.map((m) => m.metric_definition_id === sm.metric_definition_id
                                ? { ...m, prompt_override: e.target.value || null }
                                : m)
                            )}
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

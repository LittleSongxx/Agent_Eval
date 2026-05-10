import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import { DeleteOutlined, EditOutlined, PlusOutlined } from '@ant-design/icons';
import type { MetricDefinition } from '../types';
import * as api from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

const DEFAULT_REFERENCE_PROMPT = `请根据评估标准和参考答案，判断被测接口回答是否满足业务要求。

用户输入：
{user_input}

参考答案：
{reference}

评估标准：
{rubrics}

被测接口回答：
{response}

评分规则：
1. 如果回答准确覆盖参考答案中的关键要求，并符合评估标准，给 0.8 到 1.0 分。
2. 如果回答大体正确但遗漏部分关键点，给 0.5 到 0.79 分。
3. 如果回答明显偏题、违反评估标准，或缺少核心结论，给 0 到 0.49 分。
4. 理由必须指出具体满足项和缺失项。`;

const metricTypeOptions = [
  { label: '0~1 数值评分 numeric（推荐）', value: 'numeric' },
  { label: '标签判断 discrete', value: 'discrete' },
  { label: '0/1 通过判断 aspect_critic', value: 'aspect_critic' },
];

const categoryOptions = [
  { label: '通用/自定义', value: 'custom' },
  { label: 'RAG', value: 'rag' },
  { label: 'Agent', value: 'agent' },
  { label: '多轮对话', value: 'multi_turn' },
  { label: '通用', value: 'general' },
];

const typeLabelMap: Record<string, string> = {
  numeric: '数值评分',
  discrete: '离散判断',
  aspect_critic: '维度判断',
};

const metricTypeHelp: Record<string, { message: string; description: string; type: 'info' | 'warning' | 'success' }> = {
  numeric: {
    type: 'success',
    message: '适合有分档规则的业务评分',
    description: '系统要求 Judge 返回 0~1 数值分。业务提示词可以写 0.8~1.0、0.5~0.79、0~0.49 这类评分规则，报告会统计平均分和指标通过率。',
  },
  discrete: {
    type: 'info',
    message: '适合固定标签结论',
    description: '系统要求 Judge 只能返回允许标签之一，例如 pass/fail、yes/no。业务提示词应说明每个标签的判定条件。',
  },
  aspect_critic: {
    type: 'warning',
    message: '只适合是否满足的 0/1 判断',
    description: '系统要求 Judge 返回 0 或 1。业务提示词不要再写 0.8~1.0 这类连续分规则，否则系统口径和业务口径会冲突。',
  },
};

const MetricListPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricDefinition[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingMetric, setEditingMetric] = useState<MetricDefinition | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const selectedType = Form.useWatch('metric_type', form) || 'numeric';

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const data = await api.listMetrics();
      setMetrics(Array.isArray(data) ? data : []);
    } catch {
      message.error('加载指标失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const customCount = useMemo(() => metrics.filter((item) => !item.is_builtin).length, [metrics]);

  const openCreateModal = () => {
    form.resetFields();
    setEditingMetric(null);
    form.setFieldsValue({
      display_name: '参考答案符合度',
      name: 'reference_answer_quality',
      metric_type: 'numeric',
      category: 'custom',
      prompt: DEFAULT_REFERENCE_PROMPT,
      description: '根据参考答案和业务评估标准判断被测回答质量',
      min_score: 0,
      max_score: 1,
      allowed_values_text: 'pass,fail',
    });
    setModalOpen(true);
  };

  const openEditModal = (metric: MetricDefinition) => {
    const config = metric.config || {};
    const allowedValues = config.allowed_values || [0, 1];
    form.resetFields();
    setEditingMetric(metric);
    form.setFieldsValue({
      display_name: metric.display_name,
      name: metric.name,
      metric_type: metric.metric_type,
      category: metric.category || 'custom',
      prompt: config.prompt || config.definition || config.description || '',
      description: config.description || '',
      min_score: Array.isArray(allowedValues) ? allowedValues[0] : 0,
      max_score: Array.isArray(allowedValues) ? allowedValues[1] : 1,
      allowed_values_text: Array.isArray(allowedValues) ? allowedValues.join(',') : 'pass,fail',
    });
    setModalOpen(true);
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      const metricType = values.metric_type;
      const config =
        metricType === 'numeric'
          ? {
              prompt: values.prompt,
              description: values.description,
              allowed_values: [Number(values.min_score ?? 0), Number(values.max_score ?? 1)],
            }
          : metricType === 'discrete'
            ? {
                prompt: values.prompt,
                description: values.description,
                allowed_values: String(values.allowed_values_text || 'pass,fail')
                  .split(',')
                  .map((item) => item.trim())
                  .filter(Boolean),
              }
            : {
                definition: values.prompt,
                description: values.description,
              };

      const payload = {
        name: values.name,
        display_name: values.display_name,
        metric_type: metricType,
        category: values.category,
        config,
      };
      if (editingMetric) {
        await api.updateMetric(editingMetric.id, payload);
        message.success('指标已更新，后续新评测会使用最新配置');
      } else {
        await api.createMetric(payload);
        message.success('指标已创建，可在场景管理中选择使用');
      }
      setModalOpen(false);
      setEditingMetric(null);
      form.resetFields();
      fetchMetrics();
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      message.error(detail === 'Metric name already exists' ? '指标唯一标识已存在' : '保存指标失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (metric: MetricDefinition) => {
    try {
      await api.deleteMetric(metric.id);
      message.success('指标已删除');
      fetchMetrics();
    } catch {
      message.error('删除失败：内置指标或已被场景引用的指标不能删除');
    }
  };

  const columns = [
    {
      title: '指标',
      key: 'metric',
      width: 260,
      render: (_: unknown, record: MetricDefinition) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Text strong>{record.display_name}</Text>
            {record.is_builtin ? <Tag color="blue">内置</Tag> : <Tag color="purple">自定义</Tag>}
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.name}</Text>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'metric_type',
      key: 'metric_type',
      width: 140,
      render: (value: string) => <Tag>{typeLabelMap[value] || value}</Tag>,
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (value: string) => <Tag color="geekblue">{value || '-'}</Tag>,
    },
    {
      title: '评分提示/定义',
      key: 'prompt',
      render: (_: unknown, record: MetricDefinition) => {
        const prompt = record.config?.prompt || record.config?.definition || record.config?.description || '';
        return (
          <Paragraph ellipsis={{ rows: 2, expandable: true, symbol: '展开' }} style={{ marginBottom: 0 }}>
            {prompt || <Text type="secondary">-</Text>}
          </Paragraph>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      width: 190,
      render: (_: unknown, record: MetricDefinition) =>
        record.is_builtin ? (
          <Text type="secondary">不可删除</Text>
        ) : (
          <Space>
            <Button size="small" icon={<EditOutlined />} onClick={() => openEditModal(record)}>编辑</Button>
            <Popconfirm title="确认删除此自定义指标？" onConfirm={() => handleDelete(record)}>
              <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
            </Popconfirm>
          </Space>
        ),
    },
  ];

  return (
    <Spin spinning={loading}>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <Title level={4} style={{ marginBottom: 4 }}>指标管理</Title>
          <Text type="secondary">管理可复用的评测指标和默认评分规则。场景管理可按业务需要覆盖这里的默认规则。</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
          新建自定义指标
        </Button>
      </Space>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="这里维护指标默认评分规则"
        description={`默认评分规则适合放稳定、通用、可复用的判断标准。数值评分会自动参与平均分、通过率、失败样本和基线对比统计。当前共有 ${metrics.length} 个指标，其中 ${customCount} 个自定义指标。提示词支持 {user_input}、{reference}、{rubrics}、{response} 等变量。`}
      />

      <Card>
        <Table
          rowKey="id"
          columns={columns}
          dataSource={metrics}
          pagination={{ pageSize: 10, showTotal: (total) => `共 ${total} 个指标` }}
        />
      </Card>

      <Modal
        open={modalOpen}
        title={editingMetric ? '编辑自定义指标' : '新建自定义指标'}
        width={920}
        onCancel={() => {
          setModalOpen(false);
          setEditingMetric(null);
        }}
        onOk={handleSubmit}
        confirmLoading={submitting}
        destroyOnClose
      >
        <Form form={form} layout="vertical">
          <Space align="start" style={{ width: '100%' }} size={16}>
            <Form.Item
              name="display_name"
              label="展示名称"
              rules={[{ required: true, message: '请填写展示名称' }]}
              style={{ width: 260 }}
            >
              <Input placeholder="参考答案符合度" />
            </Form.Item>
            <Form.Item
              name="name"
              label="唯一标识"
              rules={[
                { required: true, message: '请填写唯一标识' },
                { pattern: /^[a-z][a-z0-9_]*$/, message: '仅支持小写字母、数字、下划线，且以字母开头' },
              ]}
              style={{ width: 260 }}
            >
              <Input placeholder="reference_answer_quality" />
            </Form.Item>
            <Form.Item name="metric_type" label="指标类型" rules={[{ required: true }]} style={{ width: 220 }}>
              <Select options={metricTypeOptions} />
            </Form.Item>
          </Space>

          <Alert
            type={metricTypeHelp[selectedType]?.type || 'info'}
            showIcon
            style={{ marginBottom: 16 }}
            message={metricTypeHelp[selectedType]?.message}
            description={metricTypeHelp[selectedType]?.description}
          />

          <Space align="start" style={{ width: '100%' }} size={16}>
            <Form.Item name="category" label="适用分类" rules={[{ required: true }]} style={{ width: 220 }}>
              <Select options={categoryOptions} />
            </Form.Item>
            <Form.Item name="description" label="说明" style={{ flex: 1 }}>
              <Input placeholder="用于报告展示和业务理解" />
            </Form.Item>
          </Space>

          {selectedType === 'numeric' && (
            <Space align="start" size={16}>
              <Form.Item name="min_score" label="最低分" rules={[{ required: true }]} style={{ width: 140 }}>
                <Input type="number" />
              </Form.Item>
              <Form.Item name="max_score" label="最高分" rules={[{ required: true }]} style={{ width: 140 }}>
                <Input type="number" />
              </Form.Item>
            </Space>
          )}

          {selectedType === 'discrete' && (
            <Form.Item name="allowed_values_text" label="允许标签">
              <Input placeholder="pass,fail" />
            </Form.Item>
          )}

          <Form.Item
            name="prompt"
            label={selectedType === 'aspect_critic' ? '默认判断定义' : '默认评分规则'}
            extra="这是指标的默认规则；在场景管理里留空时会使用这里的规则，填写场景覆盖规则时会被替代。"
            rules={[{ required: true, message: '请填写默认评分规则' }]}
          >
            <TextArea autoSize={{ minRows: 12, maxRows: 20 }} />
          </Form.Item>

          <Alert
            type="success"
            showIcon
            message="变量说明"
            description="提示词支持使用 {user_input}、{reference}、{rubrics}、{response} 等字段变量。接口实时评测会先调用被测接口得到 response，再交给该指标评分。"
          />
        </Form>
      </Modal>
    </Spin>
  );
};

export default MetricListPage;

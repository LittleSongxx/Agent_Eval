import React, { useEffect, useState } from 'react';
import { Alert, Form, Input, InputNumber, Modal, Select, Space, message } from 'antd';
import type { EvalScenario, MetricDefinition } from '../../types';
import * as api from '../../services/api';

const sceneTypeOptions = [
  { label: 'RAG', value: 'rag' },
  { label: 'Agent', value: 'agent' },
  { label: '多轮对话', value: 'multi_turn' },
  { label: '通用', value: 'general' },
];

const sampleTypeOptions = [
  { label: '单轮对话', value: 'single_turn' },
  { label: '多轮对话', value: 'multi_turn' },
];

const getMetricSearchText = (metric: MetricDefinition) => [
  metric.display_name,
  metric.name,
  metric.category,
  metric.metric_type,
  metric.is_builtin ? '内置 builtin' : '自定义 custom',
  metric.config?.description,
].filter(Boolean).join(' ').toLowerCase();

interface ScenarioQuickCreateModalProps {
  open: boolean;
  metrics: MetricDefinition[];
  initialSampleType?: string;
  onCancel: () => void;
  onCreated: (scenario: EvalScenario) => void;
}

const ScenarioQuickCreateModal: React.FC<ScenarioQuickCreateModalProps> = ({
  open,
  metrics,
  initialSampleType = 'single_turn',
  onCancel,
  onCreated,
}) => {
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    form.resetFields();
    form.setFieldsValue({
      scene_type: 'general',
      sample_type: initialSampleType,
      pass_threshold: 0.7,
      metric_ids: metrics.length > 0 ? [metrics[0].id] : [],
    });
  }, [form, initialSampleType, metrics, open]);

  const submit = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      const created = await api.createScenario({
        name: values.name,
        description: values.description || '',
        scene_type: values.scene_type,
        sample_type: values.sample_type,
        metrics: (values.metric_ids || []).map((metricId: number) => ({
          metric_definition_id: metricId,
          weight: 1.0,
          pass_threshold: values.pass_threshold ?? null,
          prompt_override: null,
        })),
      });
      message.success('场景创建成功');
      onCreated(created);
      onCancel();
    } catch (error: any) {
      if (!error?.errorFields) message.error('创建场景失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="新建评测场景"
      open={open}
      onOk={submit}
      confirmLoading={submitting}
      onCancel={onCancel}
      okText="创建"
      cancelText="取消"
      width={760}
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        <Form.Item name="name" label="场景名称" rules={[{ required: true, message: '请输入场景名称' }]}>
          <Input placeholder="例如: 客服问答 RAG 评测" />
        </Form.Item>
        <Form.Item name="description" label="描述">
          <Input.TextArea rows={2} placeholder="场景描述" />
        </Form.Item>
        <Space align="start" style={{ width: '100%' }} size={16}>
          <Form.Item name="scene_type" label="场景类型" rules={[{ required: true }]} style={{ width: 180 }}>
            <Select options={sceneTypeOptions} />
          </Form.Item>
          <Form.Item name="sample_type" label="样本类型" rules={[{ required: true }]} style={{ width: 180 }}>
            <Select options={sampleTypeOptions} />
          </Form.Item>
          <Form.Item name="pass_threshold" label="默认通过阈值" style={{ width: 160 }}>
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
        </Space>
        <Form.Item name="metric_ids" label="评测指标" rules={[{ required: true, message: '请至少选择一个指标' }]}>
          <Select
            mode="multiple"
            showSearch
            optionFilterProp="searchText"
            filterOption={(input, option) =>
              String(option?.searchText || '').includes(input.trim().toLowerCase())
            }
            placeholder="选择指标"
            options={metrics.map((metric) => ({
              label: `${metric.display_name} · ${metric.name} · ${metric.category} · ${metric.metric_type}`,
              value: metric.id,
              searchText: getMetricSearchText(metric),
            }))}
          />
        </Form.Item>
        <Alert
          type="info"
          showIcon
          message="快速创建场景"
          description="这里复用场景创建的核心字段；如需配置每个指标的场景覆盖规则、权重和精细阈值，可创建后到场景管理页面编辑。"
        />
      </Form>
    </Modal>
  );
};

export default ScenarioQuickCreateModal;

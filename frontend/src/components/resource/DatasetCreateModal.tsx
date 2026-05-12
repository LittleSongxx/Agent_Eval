import React, { useEffect, useState } from 'react';
import { Button, Checkbox, Form, Input, Modal, Radio, Select, Space, message } from 'antd';
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import type { Dataset, FieldDefinition } from '../../types';
import * as api from '../../services/api';

const fieldTypeOptions = [
  { label: 'text', value: 'text' },
  { label: 'number', value: 'number' },
  { label: 'text_list', value: 'text_list' },
  { label: 'conversation', value: 'conversation' },
  { label: 'tool_call_list', value: 'tool_call_list' },
];

const singleTurnDefaults: FieldDefinition[] = [
  { name: 'user_input', type: 'text', required: true, description: '用户输入' },
  { name: 'response', type: 'text', required: true, description: 'LLM 回复' },
  { name: 'reference', type: 'text', required: false, description: '参考答案' },
  { name: 'retrieved_contexts', type: 'text_list', required: false, description: '检索上下文列表' },
];

const multiTurnDefaults: FieldDefinition[] = [
  { name: 'user_input', type: 'conversation', required: true, description: '多轮对话输入' },
  { name: 'reference', type: 'text', required: false, description: '参考答案' },
  { name: 'reference_tool_calls', type: 'tool_call_list', required: false, description: '参考工具调用' },
];

interface DatasetCreateModalProps {
  open: boolean;
  initialSampleType?: string;
  onCancel: () => void;
  onCreated: (dataset: Dataset) => void;
}

const DatasetCreateModal: React.FC<DatasetCreateModalProps> = ({
  open,
  initialSampleType = 'single_turn',
  onCancel,
  onCreated,
}) => {
  const [form] = Form.useForm();
  const [fieldSchema, setFieldSchema] = useState<FieldDefinition[]>(singleTurnDefaults);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    form.resetFields();
    form.setFieldsValue({ sample_type: initialSampleType });
    setFieldSchema(initialSampleType === 'multi_turn' ? [...multiTurnDefaults] : [...singleTurnDefaults]);
  }, [form, initialSampleType, open]);

  const handleSampleTypeChange = (value: string) => {
    setFieldSchema(value === 'multi_turn' ? [...multiTurnDefaults] : [...singleTurnDefaults]);
  };

  const updateField = (index: number, key: keyof FieldDefinition, value: any) => {
    const next = [...fieldSchema];
    (next[index] as any)[key] = value;
    setFieldSchema(next);
  };

  const submit = async () => {
    try {
      const values = await form.validateFields();
      const validFields = fieldSchema.filter((field) => field.name.trim() !== '');
      if (validFields.length === 0) {
        message.error('请至少定义一个字段');
        return;
      }
      setSubmitting(true);
      const created = await api.createDataset({
        name: values.name,
        description: values.description || '',
        sample_type: values.sample_type,
        field_schema: validFields,
      });
      message.success('数据集创建成功');
      onCreated(created);
      onCancel();
    } catch (error: any) {
      if (!error?.errorFields) message.error('创建数据集失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="新建数据集"
      open={open}
      onOk={submit}
      confirmLoading={submitting}
      onCancel={onCancel}
      okText="创建"
      cancelText="取消"
      width={720}
      destroyOnClose
    >
      <Form form={form} layout="vertical" initialValues={{ sample_type: initialSampleType }}>
        <Form.Item name="name" label="数据集名称" rules={[{ required: true, message: '请输入名称' }]}>
          <Input placeholder="例如: RAG 测试数据集" />
        </Form.Item>
        <Form.Item name="description" label="描述">
          <Input.TextArea placeholder="可选描述信息" rows={2} />
        </Form.Item>
        <Form.Item name="sample_type" label="样本类型" rules={[{ required: true }]}>
          <Radio.Group onChange={(e) => handleSampleTypeChange(e.target.value)}>
            <Radio.Button value="single_turn">单轮对话</Radio.Button>
            <Radio.Button value="multi_turn">多轮对话</Radio.Button>
          </Radio.Group>
        </Form.Item>
      </Form>

      <div style={{ marginBottom: 8 }}>字段定义</div>
      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {fieldSchema.map((field, index) => (
          <Space key={index} style={{ display: 'flex', marginBottom: 8 }} align="start">
            <Input placeholder="字段名" value={field.name} onChange={(e) => updateField(index, 'name', e.target.value)} style={{ width: 140 }} />
            <Select value={field.type} onChange={(val) => updateField(index, 'type', val)} options={fieldTypeOptions} style={{ width: 140 }} />
            <Checkbox checked={field.required} onChange={(e) => updateField(index, 'required', e.target.checked)}>必填</Checkbox>
            <Input placeholder="描述" value={field.description} onChange={(e) => updateField(index, 'description', e.target.value)} style={{ width: 160 }} />
            <Button type="text" danger icon={<MinusCircleOutlined />} onClick={() => setFieldSchema(fieldSchema.filter((_, i) => i !== index))} />
          </Space>
        ))}
      </div>
      <Button type="dashed" onClick={() => setFieldSchema([...fieldSchema, { name: '', type: 'text', required: false, description: '' }])} block icon={<PlusOutlined />} style={{ marginTop: 8 }}>
        添加字段
      </Button>
    </Modal>
  );
};

export default DatasetCreateModal;

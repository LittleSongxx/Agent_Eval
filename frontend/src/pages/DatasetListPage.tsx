import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Radio,
  Select,
  Checkbox,
  Space,
  message,
  Popconfirm,
  Tag,
  Typography,
  Spin,
} from 'antd';
import { PlusOutlined, DeleteOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { Dataset, FieldDefinition } from '../types';
import * as api from '../services/api';

const { Title } = Typography;

const fieldTypeOptions = [
  { label: 'text', value: 'text' },
  { label: 'number', value: 'number' },
  { label: 'text_list', value: 'text_list' },
  { label: 'conversation', value: 'conversation' },
  { label: 'tool_call_list', value: 'tool_call_list' },
];

const sampleTypeColorMap: Record<string, string> = {
  single_turn: 'blue',
  multi_turn: 'purple',
};

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

const DatasetListPage: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const [fieldSchema, setFieldSchema] = useState<FieldDefinition[]>([...singleTurnDefaults]);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const data = await api.listDatasets();
      setDatasets(Array.isArray(data) ? data : []);
    } catch {
      message.error('加载数据集列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleSampleTypeChange = (value: string) => {
    if (value === 'single_turn') {
      setFieldSchema([...singleTurnDefaults]);
    } else if (value === 'multi_turn') {
      setFieldSchema([...multiTurnDefaults]);
    }
  };

  const addField = () => {
    setFieldSchema([...fieldSchema, { name: '', type: 'text', required: false, description: '' }]);
  };

  const removeField = (index: number) => {
    const newSchema = [...fieldSchema];
    newSchema.splice(index, 1);
    setFieldSchema(newSchema);
  };

  const updateField = (index: number, key: keyof FieldDefinition, value: any) => {
    const newSchema = [...fieldSchema];
    (newSchema[index] as any)[key] = value;
    setFieldSchema(newSchema);
  };

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      const validFields = fieldSchema.filter(f => f.name.trim() !== '');
      if (validFields.length === 0) {
        message.error('请至少定义一个字段');
        return;
      }
      setSubmitting(true);
      await api.createDataset({
        name: values.name,
        description: values.description || '',
        sample_type: values.sample_type,
        field_schema: validFields,
      });
      message.success('数据集创建成功');
      setModalOpen(false);
      form.resetFields();
      setFieldSchema([...singleTurnDefaults]);
      fetchDatasets();
    } catch {
      message.error('创建失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.deleteDataset(id);
      message.success('删除成功');
      fetchDatasets();
    } catch {
      message.error('删除失败');
    }
  };

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Dataset) => (
        <a onClick={() => navigate(`/datasets/${record.id}`)}>{text}</a>
      ),
    },
    {
      title: '类型',
      dataIndex: 'sample_type',
      key: 'sample_type',
      width: 120,
      render: (v: string) => (
        <Tag color={sampleTypeColorMap[v] || 'default'}>{v}</Tag>
      ),
    },
    { title: '行数', dataIndex: 'row_count', key: 'row_count', width: 80 },
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
      render: (_: unknown, record: Dataset) => (
        <Popconfirm title="确认删除此数据集？" onConfirm={() => handleDelete(record.id)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <Spin spinning={loading}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          数据集管理
        </Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
          新建数据集
        </Button>
      </div>

      <Table
        rowKey="id"
        columns={columns}
        dataSource={datasets}
        pagination={false}
      />

      <Modal
        title="新建数据集"
        open={modalOpen}
        onOk={handleCreate}
        confirmLoading={submitting}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
          setFieldSchema([...singleTurnDefaults]);
        }}
        okText="创建"
        cancelText="取消"
        width={720}
      >
        <Form form={form} layout="vertical" initialValues={{ sample_type: 'single_turn' }}>
          <Form.Item
            name="name"
            label="数据集名称"
            rules={[{ required: true, message: '请输入名称' }]}
          >
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

        <div style={{ marginBottom: 8 }}>
          <Typography.Text strong>字段定义</Typography.Text>
        </div>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {fieldSchema.map((field, index) => (
            <Space key={index} style={{ display: 'flex', marginBottom: 8 }} align="start">
              <Input
                placeholder="字段名"
                value={field.name}
                onChange={(e) => updateField(index, 'name', e.target.value)}
                style={{ width: 140 }}
              />
              <Select
                value={field.type}
                onChange={(val) => updateField(index, 'type', val)}
                options={fieldTypeOptions}
                style={{ width: 140 }}
              />
              <Checkbox
                checked={field.required}
                onChange={(e) => updateField(index, 'required', e.target.checked)}
              >
                必填
              </Checkbox>
              <Input
                placeholder="描述"
                value={field.description}
                onChange={(e) => updateField(index, 'description', e.target.value)}
                style={{ width: 160 }}
              />
              <Button
                type="text"
                danger
                icon={<MinusCircleOutlined />}
                onClick={() => removeField(index)}
              />
            </Space>
          ))}
        </div>
        <Button type="dashed" onClick={addField} block icon={<PlusOutlined />} style={{ marginTop: 8 }}>
          添加字段
        </Button>
      </Modal>
    </Spin>
  );
};

export default DatasetListPage;

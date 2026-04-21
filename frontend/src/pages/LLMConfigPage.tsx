import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  InputNumber,
  Space,
  message,
  Popconfirm,
  Tag,
  Typography,
  Badge,
  Switch,
  Spin,
} from 'antd';
import { PlusOutlined, DeleteOutlined, EditOutlined, ExperimentOutlined } from '@ant-design/icons';
import type { LLMConfig, LLMConfigCreate } from '../types';
import * as api from '../services/api';

const { Title } = Typography;

const LLMConfigPage: React.FC = () => {
  const [configs, setConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<LLMConfig | null>(null);
  const [form] = Form.useForm<LLMConfigCreate>();
  const [testingId, setTestingId] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const fetchConfigs = async () => {
    setLoading(true);
    try {
      const data = await api.listLLMConfigs();
      setConfigs(Array.isArray(data) ? data : []);
    } catch {
      message.error('加载 LLM 配置失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const openCreateModal = () => {
    setEditingConfig(null);
    form.resetFields();
    form.setFieldsValue({ temperature: 0.7, max_tokens: 4096, is_default: false });
    setModalOpen(true);
  };

  const openEditModal = (record: LLMConfig) => {
    setEditingConfig(record);
    form.setFieldsValue({
      name: record.name,
      provider: record.provider,
      api_base_url: record.api_base_url,
      api_key: record.api_key,
      model_name: record.model_name,
      temperature: record.temperature,
      max_tokens: record.max_tokens,
      is_default: record.is_default,
    });
    setModalOpen(true);
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      if (editingConfig) {
        await api.updateLLMConfig(editingConfig.id, values);
        message.success('更新成功');
      } else {
        await api.createLLMConfig(values);
        message.success('创建成功');
      }
      setModalOpen(false);
      form.resetFields();
      setEditingConfig(null);
      fetchConfigs();
    } catch {
      message.error(editingConfig ? '更新失败' : '创建失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.deleteLLMConfig(id);
      message.success('删除成功');
      fetchConfigs();
    } catch {
      message.error('删除失败');
    }
  };

  const handleTest = async (id: number) => {
    setTestingId(id);
    try {
      const result = await api.testLLMConfig(id);
      if (result.success) {
        message.success(`连接测试成功${result.latency_ms ? ` (${result.latency_ms}ms)` : ''}`);
      } else {
        message.error(`连接测试失败: ${result.message}`);
      }
    } catch {
      message.error('测试请求失败');
    } finally {
      setTestingId(null);
    }
  };

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name', width: 160 },
    {
      title: '提供商',
      dataIndex: 'provider',
      key: 'provider',
      width: 120,
      render: (v: string) => <Tag color="blue">{v || '-'}</Tag>,
    },
    { title: '模型', dataIndex: 'model_name', key: 'model_name', width: 160 },
    {
      title: 'API 地址',
      dataIndex: 'api_base_url',
      key: 'api_base_url',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'is_default',
      key: 'is_default',
      width: 100,
      render: (v: boolean) =>
        v ? <Badge status="success" text="默认" /> : <Badge status="default" text="备用" />,
    },
    {
      title: '操作',
      key: 'actions',
      width: 280,
      render: (_: unknown, record: LLMConfig) => (
        <Space>
          <Button
            size="small"
            icon={<ExperimentOutlined />}
            loading={testingId === record.id}
            onClick={() => handleTest(record.id)}
          >
            测试连接
          </Button>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => openEditModal(record)}
          >
            编辑
          </Button>
          <Popconfirm title="确认删除此配置？" onConfirm={() => handleDelete(record.id)}>
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
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          LLM 配置管理
        </Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
          新增 LLM
        </Button>
      </div>

      <Table
        rowKey="id"
        columns={columns}
        dataSource={configs}
        pagination={false}
      />

      <Modal
        title={editingConfig ? '编辑 LLM 配置' : '新增 LLM 配置'}
        open={modalOpen}
        onOk={handleSubmit}
        confirmLoading={submitting}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
          setEditingConfig(null);
        }}
        okText={editingConfig ? '保存' : '创建'}
        cancelText="取消"
        width={560}
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="name"
            label="配置名称"
            rules={[{ required: true, message: '请输入名称' }]}
          >
            <Input placeholder="例如: GPT-4o 生产环境" />
          </Form.Item>
          <Form.Item name="provider" label="提供商">
            <Input placeholder="例如: openai, anthropic, minimax" />
          </Form.Item>
          <Form.Item
            name="api_base_url"
            label="API 地址"
            rules={[{ required: true, message: '请输入 API 地址' }]}
          >
            <Input placeholder="https://api.openai.com/v1" />
          </Form.Item>
          <Form.Item
            name="api_key"
            label="API Key"
            rules={[{ required: true, message: '请输入 API Key' }]}
          >
            <Input.Password placeholder="sk-..." />
          </Form.Item>
          <Form.Item
            name="model_name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="例如: gpt-4o" />
          </Form.Item>
          <Form.Item name="temperature" label="温度 (Temperature)">
            <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="max_tokens" label="最大 Token 数">
            <InputNumber min={1} max={128000} step={256} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="is_default" label="设为默认" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </Spin>
  );
};

export default LLMConfigPage;

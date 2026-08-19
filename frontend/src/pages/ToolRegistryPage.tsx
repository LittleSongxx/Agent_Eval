import React, { useEffect, useState } from 'react';
import { Button, Card, Form, Input, InputNumber, Modal, Select, Space, Switch, Table, Tag, Typography, message } from 'antd';
import { DeleteOutlined, PlusOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import * as api from '../services/api';
import type { ToolDefinition } from '../types';

const { Title, Text } = Typography;

const ToolRegistryPage: React.FC = () => {
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const refresh = async () => {
    setLoading(true);
    try { setTools(await api.listToolRegistry()); }
    catch { message.error('加载工具目录失败'); }
    finally { setLoading(false); }
  };
  useEffect(() => { refresh(); }, []);
  const submit = async () => {
    const values = await form.validateFields();
    try {
      await api.createToolRegistryItem({ ...values, parameters_schema: values.parameters_schema ? JSON.parse(values.parameters_schema) : null });
      message.success('工具已登记'); setOpen(false); form.resetFields(); refresh();
    } catch (error: any) { message.error(error?.response?.data?.detail || '登记失败，请检查 JSON Schema'); }
  };
  return <div>
    <Space style={{ marginBottom: 16 }}><Title level={4} style={{ margin: 0 }}>Tool Registry</Title><Text type="secondary">工具 Schema、风险等级、幂等与超时元数据</Text></Space>
    <Card extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>登记工具</Button>}>
      <Table rowKey="id" loading={loading} dataSource={tools} pagination={false} columns={[
        { title: '工具', dataIndex: 'name' },
        { title: '风险', dataIndex: 'risk_level', render: (v: string) => <Tag color={v === 'high' || v === 'critical' ? 'red' : v === 'medium' ? 'orange' : 'green'}>{v}</Tag> },
        { title: '副作用', dataIndex: 'has_side_effect', render: (v: boolean) => v ? '是' : '否' },
        { title: '幂等要求', dataIndex: 'idempotency_required', render: (v: boolean) => v ? '必须' : '可选' },
        { title: '超时', dataIndex: 'timeout_ms', render: (v: number | null) => v ? String(v) + ' ms' : '-' },
        { title: '状态', dataIndex: 'enabled', render: (v: boolean) => v ? <Tag color="green">启用</Tag> : <Tag>停用</Tag> },
        { title: '操作', key: 'action', render: (_: unknown, row: ToolDefinition) => <Button danger type="text" icon={<DeleteOutlined />} title="删除工具" onClick={async () => { await api.deleteToolRegistryItem(row.id); refresh(); }} /> },
      ]} />
    </Card>
    <Modal title="登记 Agent 工具" open={open} onCancel={() => setOpen(false)} onOk={submit}>
      <Form form={form} layout="vertical" initialValues={{ risk_level: 'low', enabled: true, has_side_effect: false, idempotency_required: false }}>
        <Form.Item name="name" label="工具名" rules={[{ required: true }]}><Input placeholder="query_order" /></Form.Item>
        <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="parameters_schema" label="参数 JSON Schema"><Input.TextArea rows={5} placeholder='{"type":"object","required":["order_id"]}' /></Form.Item>
        <Space>
          <Form.Item name="risk_level" label="风险等级"><Select options={['low','medium','high','critical'].map(v => ({ label: v, value: v }))} /></Form.Item>
          <Form.Item name="timeout_ms" label="超时(ms)"><InputNumber min={1} /></Form.Item>
        </Space>
        <Space>
          <Form.Item name="has_side_effect" label="有副作用" valuePropName="checked"><Switch /></Form.Item>
          <Form.Item name="idempotency_required" label="要求幂等" valuePropName="checked"><Switch /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Space>
      </Form>
    </Modal>
  </div>;
};

export default ToolRegistryPage;

import React, { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Modal,
  Popconfirm,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import { DeleteOutlined, EditOutlined, ExperimentOutlined, PlusOutlined } from '@ant-design/icons';
import type { EndpointTarget } from '../types';
import * as api from '../services/api';
import EndpointTargetModal from '../components/resource/EndpointTargetModal';

const { Title, Text, Paragraph } = Typography;

const DEFAULT_TEST_INPUT = '请用三句话介绍一下 DeepSeek，并说明它适合做哪些 AI 应用测试。';

const EndpointTargetPage: React.FC = () => {
  const [targets, setTargets] = useState<EndpointTarget[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingTarget, setEditingTarget] = useState<EndpointTarget | null>(null);
  const [testingId, setTestingId] = useState<number | null>(null);

  const fetchTargets = async () => {
    setLoading(true);
    try {
      const data = await api.listEndpointTargets();
      setTargets(Array.isArray(data) ? data : []);
    } catch {
      message.error('加载被测接口失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTargets();
  }, []);

  const openCreateModal = () => {
    setEditingTarget(null);
    setModalOpen(true);
  };

  const openEditModal = (target: EndpointTarget) => {
    setEditingTarget(target);
    setModalOpen(true);
  };

  const handleDelete = async (target: EndpointTarget) => {
    try {
      await api.deleteEndpointTarget(target.id);
      message.success('被测接口已删除');
      fetchTargets();
    } catch {
      message.error('删除失败');
    }
  };

  const handleTest = async (target: EndpointTarget) => {
    setTestingId(target.id);
    try {
      const result = await api.testEndpointTarget(target.id, {
        row_data: { user_input: target.default_test_input || DEFAULT_TEST_INPUT },
      });
      if (result.success) {
        Modal.success({
          width: 760,
          title: '接口试跑成功',
          content: (
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <div>
                <Text strong>请求体</Text>
                <pre style={{ maxHeight: 160, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(result.request_body, null, 2)}</pre>
              </div>
              <div>
                <Text strong>解析字段</Text>
                <pre style={{ maxHeight: 160, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(result.extracted_fields || {}, null, 2)}</pre>
              </div>
            </Space>
          ),
        });
      } else {
        message.error(`接口试跑失败: ${result.message}`);
      }
    } catch {
      message.error('接口试跑请求失败');
    } finally {
      setTestingId(null);
    }
  };

  const columns = [
    {
      title: '被测接口',
      key: 'name',
      width: 240,
      render: (_: unknown, record: EndpointTarget) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.description || '-'}</Text>
        </Space>
      ),
    },
    {
      title: '接口地址',
      dataIndex: 'endpoint_url',
      key: 'endpoint_url',
      ellipsis: true,
    },
    {
      title: '返回方式',
      dataIndex: 'transport_mode',
      key: 'transport_mode',
      width: 100,
      render: (value: string) => <Tag>{value === 'sse' ? 'SSE' : 'JSON'}</Tag>,
    },
    {
      title: '回答字段',
      key: 'response_path',
      width: 220,
      render: (_: unknown, record: EndpointTarget) => <Text code>{record.response_mapping?.response_path || '-'}</Text>,
    },
    {
      title: '试跑输入',
      key: 'default_test_input',
      render: (_: unknown, record: EndpointTarget) => (
        <Paragraph ellipsis={{ rows: 2, expandable: true, symbol: '展开' }} style={{ marginBottom: 0 }}>
          {record.default_test_input || '-'}
        </Paragraph>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 260,
      render: (_: unknown, record: EndpointTarget) => (
        <Space>
          <Button size="small" icon={<ExperimentOutlined />} loading={testingId === record.id} onClick={() => handleTest(record)}>
            试跑
          </Button>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEditModal(record)}>
            编辑
          </Button>
          <Popconfirm title="确认删除此被测接口？" onConfirm={() => handleDelete(record)}>
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
          <Title level={4} style={{ marginBottom: 4 }}>被测接口管理</Title>
          <Text type="secondary">保存业务接口、模型接口和字段映射，评测执行时直接选择复用，并冻结为任务快照。</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>新建被测接口</Button>
      </Space>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="推荐把 DeepSeek、业务 RAG、Agent 服务都维护成被测接口"
        description="接口配置包含请求模板和响应字段映射。新建评测任务时选择接口后会自动带出配置，历史任务仍保留当时快照。"
      />

      <Card>
        <Table rowKey="id" columns={columns} dataSource={targets} pagination={{ pageSize: 10, showTotal: (total) => `共 ${total} 个接口` }} />
      </Card>

      <EndpointTargetModal
        open={modalOpen}
        target={editingTarget}
        onCancel={() => {
          setModalOpen(false);
          setEditingTarget(null);
        }}
        onSaved={() => {
          setModalOpen(false);
          setEditingTarget(null);
          fetchTargets();
        }}
      />
    </Spin>
  );
};

export default EndpointTargetPage;

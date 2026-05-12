import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Space,
  message,
  Popconfirm,
  Tag,
  Typography,
  Badge,
  Spin,
} from 'antd';
import { PlusOutlined, DeleteOutlined, EditOutlined, ExperimentOutlined } from '@ant-design/icons';
import type { LLMConfig } from '../types';
import * as api from '../services/api';
import LLMConfigModal from '../components/resource/LLMConfigModal';

const { Title } = Typography;

const LLMConfigPage: React.FC = () => {
  const [configs, setConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<LLMConfig | null>(null);
  const [testingId, setTestingId] = useState<number | null>(null);

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
    setModalOpen(true);
  };

  const openEditModal = (record: LLMConfig) => {
    setEditingConfig(record);
    setModalOpen(true);
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

      <LLMConfigModal
        open={modalOpen}
        config={editingConfig}
        onCancel={() => {
          setModalOpen(false);
          setEditingConfig(null);
        }}
        onSaved={() => {
          setModalOpen(false);
          setEditingConfig(null);
          fetchConfigs();
        }}
      />
    </Spin>
  );
};

export default LLMConfigPage;

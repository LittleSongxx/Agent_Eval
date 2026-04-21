import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Space,
  message,
  Popconfirm,
  Upload,
  Typography,
  Tag,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  UploadOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import type { Dataset, DatasetRow, FieldDefinition } from '../types';
import * as api from '../services/api';

const { Title, Text } = Typography;

const DatasetPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [form] = Form.useForm();

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const data = await api.listDatasets();
      setDatasets(data);
    } catch {
      message.error('加载数据集列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      const fieldNames = (values.field_names as string).split(',').map((s: string) => s.trim()).filter(Boolean);
      const field_schema: FieldDefinition[] = fieldNames.map((name: string) => ({
        name,
        type: 'string',
        required: true,
        description: '',
      }));
      await api.createDataset({
        name: values.name,
        description: values.description ?? '',
        sample_type: values.sample_type ?? 'qa',
        field_schema,
      });
      message.success('数据集创建成功');
      setCreateModalOpen(false);
      form.resetFields();
      fetchDatasets();
    } catch {
      message.error('创建失败');
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

  const handleViewRows = async (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDetailModalOpen(true);
    setRowsLoading(true);
    try {
      const res = await api.listDatasetRows(dataset.id, 1, 50);
      setRows(res.items);
    } catch {
      message.error('加载数据行失败');
    } finally {
      setRowsLoading(false);
    }
  };

  const handleImport = async (datasetId: number, file: File) => {
    try {
      const result = await api.importDataset(datasetId, file);
      message.success(`成功导入 ${result.imported_count} 条数据`);
      fetchDatasets();
    } catch {
      message.error('导入失败');
    }
  };

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    {
      title: '样本类型',
      dataIndex: 'sample_type',
      key: 'sample_type',
      render: (v: string) => <Tag>{v}</Tag>,
    },
    {
      title: '字段',
      dataIndex: 'field_schema',
      key: 'field_schema',
      render: (fields: FieldDefinition[]) =>
        fields?.map((f) => <Tag key={f.name}>{f.name}</Tag>) ?? '-',
    },
    { title: '行数', dataIndex: 'row_count', key: 'row_count' },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, record: Dataset) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewRows(record)}>
            查看
          </Button>
          <Upload
            showUploadList={false}
            accept=".csv,.json,.jsonl"
            beforeUpload={(file) => {
              handleImport(record.id, file);
              return false;
            }}
          >
            <Button size="small" icon={<UploadOutlined />}>
              导入
            </Button>
          </Upload>
          <Popconfirm title="确认删除此数据集？" onConfirm={() => handleDelete(record.id)}>
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const rowColumns = selectedDataset?.field_schema?.map((f) => ({
    title: f.name,
    dataIndex: ['data', f.name],
    key: f.name,
    ellipsis: true,
  })) ?? [];

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>数据集管理</Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
          新建数据集
        </Button>
      </div>

      <Table
        rowKey="id"
        loading={loading}
        columns={columns}
        dataSource={datasets}
        pagination={false}
      />

      <Modal
        title="新建数据集"
        open={createModalOpen}
        onOk={handleCreate}
        onCancel={() => { setCreateModalOpen(false); form.resetFields(); }}
        okText="创建"
        cancelText="取消"
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="数据集名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="例如: QA测试集" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea placeholder="可选描述信息" rows={2} />
          </Form.Item>
          <Form.Item name="sample_type" label="样本类型" initialValue="qa">
            <Input placeholder="例如: qa, generation, classification" />
          </Form.Item>
          <Form.Item
            name="field_names"
            label="字段名（逗号分隔）"
            rules={[{ required: true, message: '请输入字段名' }]}
          >
            <Input placeholder="question, answer, context" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={
          <Space>
            <Text strong>{selectedDataset?.name}</Text>
            <Text type="secondary">共 {selectedDataset?.row_count ?? 0} 行</Text>
          </Space>
        }
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={900}
      >
        <Table
          rowKey="id"
          loading={rowsLoading}
          columns={rowColumns}
          dataSource={rows}
          size="small"
          scroll={{ x: 'max-content' }}
          pagination={{ pageSize: 10 }}
        />
      </Modal>
    </>
  );
};

export default DatasetPage;

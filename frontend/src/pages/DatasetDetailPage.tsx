import React, { useEffect, useState, useCallback } from 'react';
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
  Spin,
  Descriptions,
  Upload,
} from 'antd';
import {
  ArrowLeftOutlined,
  PlusOutlined,
  DeleteOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate } from 'react-router-dom';
import type { Dataset, DatasetRow, FieldDefinition, PaginatedResponse } from '../types';
import * as api from '../services/api';

const { Title, Text, Paragraph } = Typography;

const JSON_FIELD_TYPES = new Set(['conversation', 'tool_call_list', 'text_list', 'tags', 'json']);

const isJsonValue = (val: any): boolean =>
  val !== null && val !== undefined && typeof val === 'object';

const JsonPreviewModal: React.FC<{
  open: boolean;
  title: string;
  data: any;
  onClose: () => void;
}> = ({ open, title, data, onClose }) => (
  <Modal
    title={<Space><Tag color="blue">JSON</Tag>{title}</Space>}
    open={open}
    onCancel={onClose}
    footer={<Button onClick={onClose}>关闭</Button>}
    width={720}
  >
    <pre
      style={{
        background: '#f5f5f5',
        padding: 16,
        borderRadius: 8,
        maxHeight: 500,
        overflow: 'auto',
        fontSize: 13,
        lineHeight: 1.6,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}
    >
      {JSON.stringify(data, null, 2)}
    </pre>
  </Modal>
);

const DatasetDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const datasetId = Number(id);

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [loading, setLoading] = useState(false);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [addRowModalOpen, setAddRowModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [rowForm] = Form.useForm();
  const [jsonPreview, setJsonPreview] = useState<{ open: boolean; title: string; data: any }>({
    open: false, title: '', data: null,
  });

  const fetchDataset = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getDataset(datasetId);
      setDataset(data);
    } catch {
      message.error('加载数据集信息失败');
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  const fetchRows = useCallback(async () => {
    setRowsLoading(true);
    try {
      const res: PaginatedResponse<DatasetRow> = await api.listDatasetRows(
        datasetId,
        page,
        pageSize
      );
      setRows(res.items || []);
      setTotal(res.total || 0);
    } catch {
      message.error('加载数据行失败');
    } finally {
      setRowsLoading(false);
    }
  }, [datasetId, page, pageSize]);

  useEffect(() => {
    fetchDataset();
  }, [fetchDataset]);

  useEffect(() => {
    fetchRows();
  }, [fetchRows]);

  const handleAddRow = async () => {
    try {
      const values = await rowForm.validateFields();
      setSubmitting(true);
      await api.createDatasetRow(datasetId, { data: values });
      message.success('添加成功');
      setAddRowModalOpen(false);
      rowForm.resetFields();
      fetchRows();
      fetchDataset();
    } catch {
      message.error('添加失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteRow = async (rowId: number) => {
    try {
      await api.deleteDatasetRow(datasetId, rowId);
      message.success('删除成功');
      fetchRows();
      fetchDataset();
    } catch {
      message.error('删除失败');
    }
  };

  const handleImportSuccess = () => {
    setImportModalOpen(false);
    fetchRows();
    fetchDataset();
  };

  const truncate = (text: any, maxLen = 100): string => {
    if (text === null || text === undefined) return '-';
    const str = typeof text === 'string' ? text : JSON.stringify(text);
    return str.length > maxLen ? str.slice(0, maxLen) + '...' : str;
  };

  const fieldSchema: FieldDefinition[] = dataset?.field_schema || [];

  const dataColumns = fieldSchema.map((field) => ({
    title: field.name,
    dataIndex: ['data', field.name],
    key: field.name,
    ellipsis: true,
    render: (val: any) => {
      if (val === null || val === undefined) return <Text type="secondary">-</Text>;
      if (JSON_FIELD_TYPES.has(field.type) || isJsonValue(val)) {
        const label = Array.isArray(val)
          ? `[${val.length} 项]`
          : typeof val === 'object'
            ? '{...}'
            : truncate(val, 40);
        return (
          <Button
            type="link"
            size="small"
            style={{ padding: 0 }}
            onClick={() => setJsonPreview({ open: true, title: field.name, data: val })}
          >
            📋 {label} 点击查看
          </Button>
        );
      }
      return truncate(val);
    },
  }));

  const columns = [
    {
      title: '#',
      dataIndex: 'row_index',
      key: 'row_index',
      width: 60,
    },
    ...dataColumns,
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (_: unknown, record: DatasetRow) => (
        <Popconfirm title="确认删除此行？" onConfirm={() => handleDeleteRow(record.id)}>
          <Button size="small" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ];

  const renderFieldInput = (field: FieldDefinition) => {
    switch (field.type) {
      case 'number':
        return <InputNumber style={{ width: '100%' }} />;
      case 'text_list':
        return <Input.TextArea rows={3} placeholder="每行一项，或输入 JSON 数组" />;
      case 'conversation':
        return (
          <Input.TextArea
            rows={4}
            placeholder='JSON 格式的对话数组，如 [{"type":"human","content":"hi"}]'
          />
        );
      case 'tool_call_list':
        return <Input.TextArea rows={3} placeholder="JSON 格式的工具调用列表" />;
      default:
        return <Input.TextArea rows={2} />;
    }
  };

  return (
    <Spin spinning={loading}>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets')}>
          返回列表
        </Button>
      </Space>

      {dataset && (
        <Descriptions
          title={
            <Space>
              <Title level={4} style={{ margin: 0 }}>
                {dataset.name}
              </Title>
              <Tag color="blue">{dataset.sample_type}</Tag>
            </Space>
          }
          style={{ marginBottom: 24 }}
          column={3}
        >
          <Descriptions.Item label="数据行数">{dataset.row_count}</Descriptions.Item>
          <Descriptions.Item label="创建时间">
            {new Date(dataset.created_at).toLocaleString('zh-CN')}
          </Descriptions.Item>
          <Descriptions.Item label="描述">{dataset.description || '-'}</Descriptions.Item>
        </Descriptions>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16, gap: 8 }}>
        <Button icon={<UploadOutlined />} onClick={() => setImportModalOpen(true)}>
          导入数据
        </Button>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddRowModalOpen(true)}>
          添加一行
        </Button>
      </div>

      <Table
        rowKey="id"
        loading={rowsLoading}
        columns={columns}
        dataSource={rows}
        scroll={{ x: 'max-content' }}
        pagination={{
          current: page,
          pageSize: pageSize,
          total: total,
          showSizeChanger: true,
          showTotal: (t) => `共 ${t} 条`,
          onChange: (p, ps) => {
            setPage(p);
            setPageSize(ps);
          },
        }}
      />

      <Modal
        title="添加一行数据"
        open={addRowModalOpen}
        onOk={handleAddRow}
        confirmLoading={submitting}
        onCancel={() => {
          setAddRowModalOpen(false);
          rowForm.resetFields();
        }}
        okText="添加"
        cancelText="取消"
        width={600}
      >
        <Form form={rowForm} layout="vertical" style={{ marginTop: 16 }}>
          {fieldSchema.map((field) => (
            <Form.Item
              key={field.name}
              name={field.name}
              label={
                <Space>
                  <span>{field.name}</span>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    ({field.type})
                  </Text>
                </Space>
              }
              rules={
                field.required ? [{ required: true, message: `请输入 ${field.name}` }] : []
              }
            >
              {renderFieldInput(field)}
            </Form.Item>
          ))}
        </Form>
      </Modal>

      <JsonPreviewModal
        open={jsonPreview.open}
        title={jsonPreview.title}
        data={jsonPreview.data}
        onClose={() => setJsonPreview({ open: false, title: '', data: null })}
      />

      <Modal
        title="导入数据"
        open={importModalOpen}
        footer={null}
        onCancel={() => setImportModalOpen(false)}
      >
        <Upload.Dragger
          accept=".csv,.json"
          showUploadList={false}
          beforeUpload={(file) => {
            (async () => {
              try {
                const result = await api.importDataset(datasetId, file as File);
                message.success(`成功导入 ${result.imported_count ?? ''} 条数据`);
                handleImportSuccess();
              } catch {
                message.error('导入失败');
              }
            })();
            return false;
          }}
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined style={{ fontSize: 32, color: '#1677ff' }} />
          </p>
          <p className="ant-upload-text">点击或拖拽文件上传</p>
          <p className="ant-upload-hint">支持 .csv 和 .json 文件</p>
        </Upload.Dragger>
      </Modal>
    </Spin>
  );
};

export default DatasetDetailPage;

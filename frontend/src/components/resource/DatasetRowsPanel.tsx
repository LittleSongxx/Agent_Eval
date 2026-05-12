import React, { useEffect, useState, useCallback } from 'react';
import {
  Button,
  Form,
  Input,
  InputNumber,
  Modal,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd';
import { DeleteOutlined, PlusOutlined, UploadOutlined } from '@ant-design/icons';
import type { Dataset, DatasetRow, FieldDefinition, PaginatedResponse } from '../../types';
import * as api from '../../services/api';

const { Text } = Typography;

const JSON_FIELD_TYPES = new Set(['conversation', 'tool_call_list', 'text_list', 'tags', 'json']);

const isJsonValue = (val: any): boolean =>
  val !== null && val !== undefined && typeof val === 'object';

const truncate = (text: any, maxLen = 80): string => {
  if (text === null || text === undefined) return '-';
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return str.length > maxLen ? `${str.slice(0, maxLen)}...` : str;
};

const getErrorMessage = (error: any, fallback: string) => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.message || JSON.stringify(item))
      .join('；');
  }
  if (error?.message) return error.message;
  return fallback;
};

const parseListText = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed : [value];
  } catch {
    return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean) || [value];
  }
};

const normalizeRowValues = (values: Record<string, any>, fields: FieldDefinition[]) => {
  const next: Record<string, any> = {};
  fields.forEach((field) => {
    const rawValue = values[field.name];
    if (rawValue === undefined || rawValue === null || rawValue === '') {
      if (field.required) next[field.name] = rawValue;
      return;
    }
    if (field.type === 'number') {
      next[field.name] = typeof rawValue === 'number' ? rawValue : Number(rawValue);
      return;
    }
    if (field.type === 'text_list' || field.type === 'tags') {
      next[field.name] = Array.isArray(rawValue) ? rawValue : parseListText(String(rawValue));
      return;
    }
    if (field.type === 'conversation' || field.type === 'tool_call_list' || field.type === 'json') {
      next[field.name] = typeof rawValue === 'string' ? JSON.parse(rawValue) : rawValue;
      return;
    }
    next[field.name] = rawValue;
  });
  return next;
};

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

interface DatasetRowsPanelProps {
  dataset: Dataset;
  compact?: boolean;
  onDatasetUpdated?: (dataset: Dataset) => void;
}

const DatasetRowsPanel: React.FC<DatasetRowsPanelProps> = ({
  dataset,
  compact = false,
  onDatasetUpdated,
}) => {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(compact ? 5 : 20);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [addRowModalOpen, setAddRowModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [jsonPreview, setJsonPreview] = useState<{ open: boolean; title: string; data: any }>({
    open: false,
    title: '',
    data: null,
  });
  const [submitting, setSubmitting] = useState(false);
  const [rowForm] = Form.useForm();

  const datasetId = dataset.id;
  const fieldSchema = dataset.field_schema || [];

  const refresh = useCallback(async () => {
    setRowsLoading(true);
    try {
      const [nextDataset, rowPage]: [Dataset, PaginatedResponse<DatasetRow>] = await Promise.all([
        api.getDataset(datasetId),
        api.listDatasetRows(datasetId, page, pageSize),
      ]);
      setRows(rowPage.items || []);
      setTotal(rowPage.total || 0);
      onDatasetUpdated?.(nextDataset);
    } catch {
      message.error('加载数据行失败');
    } finally {
      setRowsLoading(false);
    }
  }, [datasetId, onDatasetUpdated, page, pageSize]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAddRow = async () => {
    try {
      const values = await rowForm.validateFields();
      const normalizedValues = normalizeRowValues(values, fieldSchema);
      setSubmitting(true);
      await api.createDatasetRow(datasetId, { data: normalizedValues });
      message.success('添加成功');
      setAddRowModalOpen(false);
      rowForm.resetFields();
      refresh();
    } catch (error) {
      message.error(getErrorMessage(error, '添加失败，请检查字段格式'));
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteRow = async (rowId: number) => {
    try {
      await api.deleteDatasetRow(datasetId, rowId);
      message.success('删除成功');
      refresh();
    } catch {
      message.error('删除失败');
    }
  };

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
            {label} 查看
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

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
        <Space wrap>
          <Tag color={dataset.row_count > 0 ? 'green' : 'orange'}>数据行 {dataset.row_count}</Tag>
          <Tag>字段 {fieldSchema.length}</Tag>
        </Space>
        <Space>
          <Button icon={<UploadOutlined />} onClick={() => setImportModalOpen(true)}>
            导入数据
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddRowModalOpen(true)}>
            添加一行
          </Button>
        </Space>
      </Space>

      <Table
        rowKey="id"
        size={compact ? 'small' : 'middle'}
        loading={rowsLoading}
        columns={columns}
        dataSource={rows}
        scroll={{ x: 'max-content' }}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: !compact,
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
        width={640}
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
              rules={field.required ? [{ required: true, message: `请输入 ${field.name}` }] : []}
            >
              {renderFieldInput(field)}
            </Form.Item>
          ))}
        </Form>
      </Modal>

      <Modal
        title={<Space><Tag color="blue">JSON</Tag>{jsonPreview.title}</Space>}
        open={jsonPreview.open}
        onCancel={() => setJsonPreview({ open: false, title: '', data: null })}
        footer={<Button onClick={() => setJsonPreview({ open: false, title: '', data: null })}>关闭</Button>}
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
          {JSON.stringify(jsonPreview.data, null, 2)}
        </pre>
      </Modal>

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
                const importedCount = result.imported_count ?? 0;
                const skippedDuplicates = result.skipped_duplicates ?? 0;
                message.success(
                  `成功导入 ${importedCount} 条数据${skippedDuplicates ? `，去重跳过 ${skippedDuplicates} 条` : ''}`
                );
                setImportModalOpen(false);
                refresh();
              } catch (error) {
                message.error(getErrorMessage(error, '导入失败，请检查文件格式和字段内容'));
              }
            })();
            return false;
          }}
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined style={{ fontSize: 32, color: '#1677ff' }} />
          </p>
          <p className="ant-upload-text">点击或拖拽文件上传</p>
          <p className="ant-upload-hint">支持 .csv 和 .json 文件，导入时会自动去重</p>
        </Upload.Dragger>
      </Modal>
    </Space>
  );
};

export default DatasetRowsPanel;

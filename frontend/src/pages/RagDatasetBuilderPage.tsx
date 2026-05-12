import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd';
import {
  ArrowLeftOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type {
  LLMConfig,
  PaginatedResponse,
  RagDatasetChunk,
  RagDatasetJob,
  RagDatasetSample,
} from '../types';
import * as api from '../services/api';

const { Title, Text, Paragraph } = Typography;

const statusColorMap: Record<string, string> = {
  draft: 'default',
  running: 'processing',
  completed: 'success',
  partial: 'warning',
  failed: 'error',
};

const chunkQualityColorMap: Record<string, string> = {
  good: 'success',
  medium: 'processing',
  low: 'warning',
  filtered: 'default',
};

const getErrorMessage = (error: any, fallback: string) => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((item) => item?.msg || JSON.stringify(item)).join('; ');
  }
  return fallback;
};

// Architecture note:
// This page currently exposes the document-upload implementation of test data
// generation. The RAG naming is historical and should not become the long-term
// product boundary. When the platform adds a unified test-data ingestion layer,
// this flow should move behind a generic "data source" entry alongside API
// sampling, log replay, and manual-label imports.
const RagDatasetBuilderPage: React.FC = () => {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<RagDatasetJob[]>([]);
  const [llmConfigs, setLlmConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [selectedJob, setSelectedJob] = useState<RagDatasetJob | null>(null);
  const [chunkItems, setChunkItems] = useState<RagDatasetChunk[]>([]);
  const [samples, setSamples] = useState<RagDatasetSample[]>([]);
  const [samplesTotal, setSamplesTotal] = useState(0);
  const [samplePage, setSamplePage] = useState(1);
  const [samplePageSize, setSamplePageSize] = useState(20);
  const [sampleStatus, setSampleStatus] = useState<string | undefined>(undefined);
  const [chunkPreview, setChunkPreview] = useState<RagDatasetChunk | null>(null);
  const [sampleChunkPreview, setSampleChunkPreview] = useState<RagDatasetChunk[]>([]);
  const [sampleChunkPreviewTitle, setSampleChunkPreviewTitle] = useState('');
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [jobForm] = Form.useForm();

  const fetchJobs = useCallback(async () => {
    setLoading(true);
    try {
      const [jobData, llmData] = await Promise.all([api.listRagDatasetJobs(), api.listLLMConfigs()]);
      const items: RagDatasetJob[] = Array.isArray(jobData) ? jobData : [];
      setJobs(items);
      setLlmConfigs(Array.isArray(llmData) ? llmData : []);
      if (!selectedJobId && items.length > 0) {
        setSelectedJobId(items[0].id);
      }
    } catch {
      message.error('加载文档生成测试数据任务失败');
    } finally {
      setLoading(false);
    }
  }, [selectedJobId]);

  const fetchJobDetail = useCallback(async () => {
    if (!selectedJobId) {
      setSelectedJob(null);
      setChunkItems([]);
      setSamples([]);
      setSamplesTotal(0);
      return;
    }
    setDetailLoading(true);
    try {
      const [job, chunkResp, sampleResp]: [RagDatasetJob, { items: RagDatasetChunk[] }, PaginatedResponse<RagDatasetSample>] =
        await Promise.all([
          api.getRagDatasetJob(selectedJobId),
          api.listRagDatasetJobChunks(selectedJobId),
          api.listRagDatasetJobSamples(selectedJobId, samplePage, samplePageSize, sampleStatus),
        ]);
      setSelectedJob(job);
      setChunkItems(chunkResp.items || []);
      setSamples(sampleResp.items || []);
      setSamplesTotal(sampleResp.total || 0);
    } catch {
      message.error('加载任务详情失败');
    } finally {
      setDetailLoading(false);
    }
  }, [selectedJobId, samplePage, samplePageSize, sampleStatus]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  useEffect(() => {
    fetchJobDetail();
  }, [fetchJobDetail]);

  useEffect(() => {
    if (!selectedJob) return;
    if (!['running'].includes(selectedJob.status)) return;
    const timer = window.setInterval(() => {
      fetchJobs();
      fetchJobDetail();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [selectedJob, fetchJobDetail, fetchJobs]);

  const handleCreateJob = async () => {
    try {
      const values = await jobForm.validateFields();
      setSubmitting(true);
      const job = await api.createRagDatasetJob({
        name: values.name,
        description: values.description || '',
        question_llm_config_id: values.question_llm_config_id,
        question_count_mode: values.question_count_mode,
        requested_question_count:
          values.question_count_mode === 'custom' ? values.requested_question_count : null,
      });
      message.success('已创建文档生成测试数据任务');
      setSelectedJobId(job.id);
      setPendingFiles([]);
      jobForm.resetFields();
      fetchJobs();
    } catch (error: any) {
      if (!error?.errorFields) {
        message.error(getErrorMessage(error, '创建任务失败'));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleUploadDocuments = async () => {
    if (!selectedJobId || pendingFiles.length === 0) {
      message.warning('请先选择任务并添加文档');
      return;
    }
    setUploading(true);
    try {
      await api.uploadRagDatasetJobDocuments(selectedJobId, pendingFiles);
      message.success(`已上传 ${pendingFiles.length} 个文档`);
      setPendingFiles([]);
      fetchJobs();
      fetchJobDetail();
    } catch (error: any) {
      message.error(getErrorMessage(error, '文档上传失败'));
    } finally {
      setUploading(false);
    }
  };

  const handleStart = async () => {
    if (!selectedJobId) return;
    try {
      await api.startRagDatasetJob(selectedJobId);
      message.success('已启动生成任务');
      fetchJobDetail();
    } catch (error: any) {
      message.error(getErrorMessage(error, '启动生成任务失败'));
    }
  };

  const handleRetryFailed = async () => {
    if (!selectedJobId) return;
    try {
      await api.retryFailedRagDatasetJob(selectedJobId);
      message.success('已启动失败重试');
      fetchJobDetail();
    } catch (error: any) {
      message.error(getErrorMessage(error, '启动失败重试失败'));
    }
  };

  const chunkColumns = [
    { title: '分片', dataIndex: 'chunk_key', key: 'chunk_key', width: 180 },
    {
      title: '文档',
      key: 'document_id',
      width: 160,
      render: (_: unknown, record: RagDatasetChunk) =>
        selectedJob?.documents.find((item) => item.id === record.document_id)?.filename || `文档 #${record.document_id}`,
    },
    { title: '建议条数', dataIndex: 'suggested_question_count', key: 'suggested', width: 90 },
    { title: '分配条数', dataIndex: 'allocated_question_count', key: 'allocated', width: 90 },
    {
      title: '质量',
      key: 'quality',
      width: 120,
      render: (_: unknown, record: RagDatasetChunk) => (
        <Tag color={chunkQualityColorMap[record.quality_label] || 'default'}>
          {record.quality_label} · {record.quality_score}
        </Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'generation_status',
      key: 'generation_status',
      width: 110,
      render: (value: string) => <Tag color={statusColorMap[value] || 'default'}>{value}</Tag>,
    },
    {
      title: '内容预览',
      key: 'preview',
      render: (_: unknown, record: RagDatasetChunk) => (
        <Button type="link" size="small" onClick={() => setChunkPreview(record)}>
          查看 chunk
        </Button>
      ),
    },
  ];

  const sampleColumns = [
    { title: '#', dataIndex: 'id', key: 'id', width: 70 },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (value: string) => <Tag color={statusColorMap[value] || 'default'}>{value}</Tag>,
    },
    {
      title: '问题',
      dataIndex: 'question',
      key: 'question',
      width: 280,
      ellipsis: true,
    },
    {
      title: '标准答案',
      dataIndex: 'reference',
      key: 'reference',
      width: 280,
      ellipsis: true,
    },
    {
      title: '引用 chunk',
      key: 'source_chunks',
      width: 130,
      render: (_: unknown, record: RagDatasetSample) => {
        const relatedChunks = chunkItems.filter((chunk) =>
          record.source_chunk_ids?.includes(chunk.chunk_key)
        );
        return (
          <Button
            type="link"
            size="small"
            disabled={relatedChunks.length === 0}
            onClick={() => {
              setSampleChunkPreview(relatedChunks);
              setSampleChunkPreviewTitle(`样本 #${record.id} 引用 chunk`);
            }}
          >
            查看 ({relatedChunks.length})
          </Button>
        );
      },
    },
    {
      title: '失败原因',
      dataIndex: 'error_message',
      key: 'error_message',
      width: 220,
      ellipsis: true,
      render: (value: string | null | undefined) => value || <Text type="secondary">-</Text>,
    },
  ];

  const selectedJobRunning = selectedJob?.status === 'running';
  const canViewDataset = !!selectedJob?.dataset_id;

  const uploadFileList = useMemo(
    () =>
      pendingFiles.map((file) => ({
        uid: file.name,
        name: file.name,
        status: 'done' as const,
      })),
    [pendingFiles]
  );

  const chunkPreviewLabel = useMemo(() => {
    if (sampleChunkPreview.length > 0) {
      return sampleChunkPreviewTitle;
    }
    if (chunkPreview) {
      return `Chunk 预览 · ${chunkPreview.chunk_key}`;
    }
    return 'Chunk 预览';
  }, [chunkPreview, sampleChunkPreview, sampleChunkPreviewTitle]);

  return (
    <Spin spinning={loading}>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets')}>
          返回数据集
        </Button>
        <Title level={4} style={{ margin: 0 }}>
          文档生成测试数据
        </Title>
      </Space>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="上传知识库文档，自动解析分片并生成测试数据（问题 + 参考答案），同步为普通评测数据集。"
      />

      <div style={{ display: 'grid', gridTemplateColumns: '360px minmax(0, 1fr)', gap: 16, alignItems: 'start' }}>
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Card title="新建生成任务">
            <Form
              form={jobForm}
              layout="vertical"
              initialValues={{
                question_count_mode: 'auto',
              }}
            >
              <Form.Item name="name" label="任务名称" rules={[{ required: true, message: '请输入任务名称' }]}>
                <Input placeholder="例如：员工手册文档测试数据生成" />
              </Form.Item>
              <Form.Item name="description" label="描述">
                <Input.TextArea rows={2} placeholder="可选说明" />
              </Form.Item>
              <Form.Item name="question_llm_config_id" label="生成模型" rules={[{ required: true, message: '请选择生成模型' }]}>
                <Select
                  placeholder="选择用于生成数据和标准答案的模型"
                  options={llmConfigs.map((item) => ({ label: item.name, value: item.id }))}
                />
              </Form.Item>
              <Form.Item name="question_count_mode" label="数量策略" rules={[{ required: true }]}>
                <Radio.Group>
                  <Radio.Button value="auto">自动建议</Radio.Button>
                  <Radio.Button value="custom">自定义总数</Radio.Button>
                </Radio.Group>
              </Form.Item>
              <Form.Item noStyle shouldUpdate={(prev, next) => prev.question_count_mode !== next.question_count_mode}>
                {({ getFieldValue }) =>
                  getFieldValue('question_count_mode') === 'custom' ? (
                    <Form.Item
                      name="requested_question_count"
                      label="总条数"
                      rules={[{ required: true, message: '请输入总条数' }]}
                    >
                      <InputNumber min={1} max={200} style={{ width: '100%' }} />
                    </Form.Item>
                  ) : null
                }
              </Form.Item>
            </Form>
            <Button type="primary" block loading={submitting} onClick={handleCreateJob}>
              创建任务
            </Button>
          </Card>

          <Card title="任务列表">
            <Table
              rowKey="id"
              size="small"
              pagination={false}
              dataSource={jobs}
              rowClassName={(record) => (record.id === selectedJobId ? 'ant-table-row-selected' : '')}
              onRow={(record) => ({ onClick: () => setSelectedJobId(record.id) })}
              columns={[
                { title: '名称', dataIndex: 'name', key: 'name', ellipsis: true },
                {
                  title: '状态',
                  dataIndex: 'status',
                  key: 'status',
                  width: 100,
                  render: (value: string) => <Tag color={statusColorMap[value] || 'default'}>{value}</Tag>,
                },
              ]}
            />
          </Card>
        </Space>

        <Spin spinning={detailLoading}>
          {selectedJob ? (
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Card
                title={selectedJob.name}
                extra={
                  <Space>
                    <Button
                      icon={<PlayCircleOutlined />}
                      type="primary"
                      onClick={handleStart}
                      disabled={selectedJobRunning || selectedJob.documents.length === 0}
                    >
                      开始生成
                    </Button>
                    <Button icon={<ReloadOutlined />} onClick={handleRetryFailed} disabled={selectedJobRunning}>
                      重试失败项
                    </Button>
                    {canViewDataset && (
                      <Button icon={<DatabaseOutlined />} onClick={() => navigate(`/datasets/${selectedJob.dataset_id}`)}>
                        查看数据集
                      </Button>
                    )}
                  </Space>
                }
              >
                <Descriptions column={4} size="small">
                  <Descriptions.Item label="状态">
                    <Tag color={statusColorMap[selectedJob.status] || 'default'}>{selectedJob.status}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="文档数">{selectedJob.total_documents}</Descriptions.Item>
                  <Descriptions.Item label="分片数">{selectedJob.total_chunks}</Descriptions.Item>
                  <Descriptions.Item label="建议条数">{selectedJob.suggested_question_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="总样本">{selectedJob.total_samples}</Descriptions.Item>
                  <Descriptions.Item label="成功">{selectedJob.completed_samples}</Descriptions.Item>
                  <Descriptions.Item label="失败">{selectedJob.failed_samples}</Descriptions.Item>
                </Descriptions>
                {selectedJob.error_message && (
                  <Alert type="error" showIcon style={{ marginTop: 16 }} message={selectedJob.error_message} />
                )}
              </Card>

              <Card title="上传知识库文档" extra={<Text type="secondary">支持 txt / md / csv / json，docx / pdf 在本地安装依赖后可用</Text>}>
                <Upload
                  multiple
                  beforeUpload={(file) => {
                    setPendingFiles((prev) => [...prev, file]);
                    return false;
                  }}
                  fileList={uploadFileList as any}
                  onRemove={(file) => {
                    setPendingFiles((prev) => prev.filter((item) => item.name !== file.name));
                  }}
                >
                  <Button icon={<UploadOutlined />}>选择文档</Button>
                </Upload>
                <Space style={{ marginTop: 12 }}>
                  <Button type="primary" loading={uploading} onClick={handleUploadDocuments} disabled={pendingFiles.length === 0}>
                    上传到当前任务
                  </Button>
                  <Text type="secondary">当前已选 {pendingFiles.length} 个文件</Text>
                </Space>
              </Card>

              <Card title="文档分片预览" extra={<Text type="secondary">点击“查看 chunk”可预览内容</Text>}>
                <Table rowKey="id" size="small" dataSource={chunkItems} columns={chunkColumns} pagination={{ pageSize: 8 }} />
              </Card>

              <Card
                title="生成样本"
                extra={
                  <Space>
                    <Select
                      allowClear
                      placeholder="状态筛选"
                      style={{ width: 160 }}
                      value={sampleStatus}
                      onChange={(value) => {
                        setSampleStatus(value);
                        setSamplePage(1);
                      }}
                      options={[
                        { label: 'completed', value: 'completed' },
                        { label: 'failed', value: 'failed' },
                        { label: 'pending', value: 'pending' },
                      ]}
                    />
                  </Space>
                }
              >
                <Table
                  rowKey="id"
                  size="small"
                  dataSource={samples}
                  columns={sampleColumns}
                  scroll={{ x: 'max-content' }}
                  pagination={{
                    current: samplePage,
                    pageSize: samplePageSize,
                    total: samplesTotal,
                    showSizeChanger: true,
                    onChange: (page, pageSize) => {
                      setSamplePage(page);
                      setSamplePageSize(pageSize);
                    },
                  }}
                />
              </Card>

              <Card title="任务日志">
                <pre
                  style={{
                    margin: 0,
                    maxHeight: 240,
                    overflow: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontSize: 12,
                    background: '#fafafa',
                    padding: 12,
                    borderRadius: 6,
                  }}
                >
                  {selectedJob.logs || '暂无日志'}
                </pre>
              </Card>
            </Space>
          ) : (
            <Card>
              <Text type="secondary">左侧创建或选择一个任务开始。</Text>
            </Card>
          )}
        </Spin>
      </div>

      <Modal
        title={chunkPreviewLabel}
        open={!!chunkPreview || sampleChunkPreview.length > 0}
        onCancel={() => {
          setChunkPreview(null);
          setSampleChunkPreview([]);
          setSampleChunkPreviewTitle('');
        }}
        footer={
          <Button
            onClick={() => {
              setChunkPreview(null);
              setSampleChunkPreview([]);
              setSampleChunkPreviewTitle('');
            }}
          >
            关闭
          </Button>
        }
        width={820}
      >
        {sampleChunkPreview.length > 0 ? (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            {sampleChunkPreview.map((item) => (
              <Card
                key={item.id}
                size="small"
                title={item.chunk_key}
                extra={<Text type="secondary">{item.char_count} chars</Text>}
              >
                <Descriptions size="small" column={3} style={{ marginBottom: 12 }}>
                  <Descriptions.Item label="建议条数">{item.suggested_question_count}</Descriptions.Item>
                  <Descriptions.Item label="分配条数">{item.allocated_question_count}</Descriptions.Item>
                  <Descriptions.Item label="状态">{item.generation_status}</Descriptions.Item>
                </Descriptions>
                <pre
                  style={{
                    margin: 0,
                    maxHeight: 220,
                    overflow: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontSize: 13,
                    lineHeight: 1.6,
                    background: '#fafafa',
                    padding: 12,
                    borderRadius: 6,
                  }}
                >
                  {item.content}
                </pre>
              </Card>
            ))}
          </Space>
        ) : chunkPreview ? (
          <>
            <Descriptions size="small" column={3} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="建议条数">{chunkPreview.suggested_question_count}</Descriptions.Item>
              <Descriptions.Item label="分配条数">{chunkPreview.allocated_question_count}</Descriptions.Item>
              <Descriptions.Item label="状态">{chunkPreview.generation_status}</Descriptions.Item>
              <Descriptions.Item label="质量">
                <Tag color={chunkQualityColorMap[chunkPreview.quality_label] || 'default'}>
                  {chunkPreview.quality_label} · {chunkPreview.quality_score}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
            {chunkPreview.quality_reasons?.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>质量判断：</Text>
                <Space size={[4, 4]} wrap>
                  {chunkPreview.quality_reasons.map((item) => (
                    <Tag key={item}>{item}</Tag>
                  ))}
                </Space>
              </div>
            )}
            {chunkPreview.generation_error && (
              <Alert type="error" showIcon style={{ marginBottom: 16 }} message={chunkPreview.generation_error} />
            )}
            <pre
              style={{
                margin: 0,
                maxHeight: 480,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontSize: 13,
                lineHeight: 1.6,
                background: '#fafafa',
                padding: 12,
                borderRadius: 6,
              }}
            >
              {chunkPreview.content}
            </pre>
          </>
        ) : null}
      </Modal>
    </Spin>
  );
};

export default RagDatasetBuilderPage;

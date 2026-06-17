import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExperimentOutlined,
  MinusCircleOutlined,
  PlayCircleOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import type { BlindTestRowResult, BlindTestSummary, BlindTestTask, Dataset, LLMConfig, PaginatedResponse } from '../types';
import * as api from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

const statusColorMap: Record<string, string> = {
  pending: 'blue',
  running: 'orange',
  completed: 'green',
  failed: 'red',
  cancelled: 'default',
};

const DEFAULT_ENDPOINT_BODY = '{\n  "question": "{{question}}"\n}';
const DEFAULT_BLIND_TEST_ENDPOINT_URL = 'https://example.com/chat/completions';
const DEFAULT_BLIND_TEST_AUTHORIZATION = 'Bearer xxx';
const DEFAULT_BLIND_TEST_HEADERS = '{\n  "origin": "https://example.com",\n  "X-App-Id": "xxx",\n  "referer": "https://example.com/"\n}';
const DEFAULT_BLIND_TEST_BODY = '{\n  "question": "{{question}}",\n  "kb_codes": [],\n  "payload": {\n    "files": []\n  }\n}';

type TargetFieldPrefix = 'target_a' | 'target_b';

const BlindTestPage: React.FC = () => {
  const [tasks, setTasks] = useState<BlindTestTask[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [llmConfigs, setLLMConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [detailTask, setDetailTask] = useState<BlindTestTask | null>(null);
  const [detailSummary, setDetailSummary] = useState<BlindTestSummary | null>(null);
  const [detailRows, setDetailRows] = useState<BlindTestRowResult[]>([]);
  const [detailRowsTotal, setDetailRowsTotal] = useState(0);
  const [detailPage, setDetailPage] = useState(1);
  const [detailPageSize, setDetailPageSize] = useState(10);
  const [detailStatus, setDetailStatus] = useState<string | undefined>(undefined);
  const [judgeRow, setJudgeRow] = useState<BlindTestRowResult | null>(null);
  const [voteNote, setVoteNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [testingTarget, setTestingTarget] = useState<TargetFieldPrefix | null>(null);
  const [form] = Form.useForm();
  const [targetTestQuestion, setTargetTestQuestion] = useState<Record<TargetFieldPrefix, string>>({
    target_a: '五一我是去广东玩好还是去重庆玩比较好，给一个建议',
    target_b: '五一我是去广东玩好还是去重庆玩比较好，给一个建议',
  });

  const fetchBaseData = useCallback(async () => {
    setLoading(true);
    try {
      const [taskData, datasetData, llmData] = await Promise.all([
        api.listBlindTests(),
        api.listDatasets(),
        api.listLLMConfigs(),
      ]);
      setTasks(Array.isArray(taskData) ? taskData : []);
      setDatasets(Array.isArray(datasetData) ? datasetData : []);
      setLLMConfigs(Array.isArray(llmData) ? llmData : []);
    } catch {
      message.error('加载人工盲测数据失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBaseData();
  }, [fetchBaseData]);

  useEffect(() => {
    if (!tasks.some((task) => task.status === 'pending' || task.status === 'running')) {
      return;
    }
    const timer = setInterval(fetchBaseData, 1200);
    return () => clearInterval(timer);
  }, [tasks, fetchBaseData]);

  const loadTaskDetail = useCallback(async (taskId: number, nextPage = detailPage, nextPageSize = detailPageSize, nextStatus = detailStatus) => {
    setDetailLoading(true);
    try {
      const [task, summary, rows] = await Promise.all([
        api.getBlindTest(taskId),
        api.getBlindTestSummary(taskId),
        api.getBlindTestRows(taskId, nextPage, nextPageSize, nextStatus),
      ]);
      setDetailTask(task);
      setDetailSummary(summary);
      setDetailRows((rows as PaginatedResponse<BlindTestRowResult>).items || []);
      setDetailRowsTotal((rows as PaginatedResponse<BlindTestRowResult>).total || 0);
    } catch {
      message.error('加载盲测任务详情失败');
    } finally {
      setDetailLoading(false);
    }
  }, [detailPage, detailPageSize, detailStatus]);

  useEffect(() => {
    if (!detailTask) return;
    loadTaskDetail(detailTask.id, detailPage, detailPageSize, detailStatus);
  }, [detailTask?.id, detailPage, detailPageSize, detailStatus, loadTaskDetail]);

  useEffect(() => {
    if (!detailTask || !['pending', 'running'].includes(detailTask.status)) return;
    const timer = setInterval(() => {
      loadTaskDetail(detailTask.id, detailPage, detailPageSize, detailStatus);
    }, 1200);
    return () => clearInterval(timer);
  }, [detailTask, detailPage, detailPageSize, detailStatus, loadTaskDetail]);

  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedDataset = useMemo(
    () => datasets.find((item) => item.id === selectedDatasetId),
    [datasets, selectedDatasetId],
  );

  const datasetHasUserInput = useMemo(() => {
    const fields = new Set((selectedDataset?.field_schema || []).map((field) => field.name));
    return fields.has('user_input');
  }, [selectedDataset]);

  const handleTestTarget = async (prefix: TargetFieldPrefix) => {
    try {
      const targetType = form.getFieldValue([prefix, 'target_type']) || 'llm_config';
      const fieldPaths =
        targetType === 'llm_config'
          ? [[prefix, 'name'], [prefix, 'llm_config_id']]
          : [[prefix, 'name'], [prefix, 'endpoint_url'], [prefix, 'request_body_template']];
      await form.validateFields(fieldPaths);
      setTestingTarget(prefix);
      const target = {
        target_type: targetType,
        name: form.getFieldValue([prefix, 'name']),
        llm_config_id: form.getFieldValue([prefix, 'llm_config_id']),
        endpoint_url: form.getFieldValue([prefix, 'endpoint_url']),
        transport_mode: form.getFieldValue([prefix, 'transport_mode']) || 'json',
        authorization: form.getFieldValue([prefix, 'authorization']) || '',
        extra_headers: form.getFieldValue([prefix, 'extra_headers']) || '{}',
        request_body_template: form.getFieldValue([prefix, 'request_body_template']) || DEFAULT_ENDPOINT_BODY,
        response_mode: 'answer_only',
      };
      const result = await api.testBlindTestTarget({
        target,
        test_question: targetTestQuestion[prefix],
      });
      if (result.success) {
        Modal.success({
          width: 720,
          title: `${prefix === 'target_a' ? '对比对象 A' : '对比对象 B'} 测试成功`,
          content: (
            <div>
              <div style={{ marginBottom: 8 }}>测试问题：{targetTestQuestion[prefix]}</div>
              <pre style={{ maxHeight: 280, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                {result.answer_preview || '接口可达，但未返回可展示的回答'}
              </pre>
            </div>
          ),
        });
      } else {
        message.error(`测试失败: ${result.message}`);
      }
    } catch {
      message.error('请先补全当前对比对象的必填配置');
    } finally {
      setTestingTarget(null);
    }
  };

  const renderTargetFields = (prefix: TargetFieldPrefix, title: string) => {
    const targetType = Form.useWatch([prefix, 'target_type'], form) || 'llm_config';
    return (
      <Card size="small" title={title} style={{ marginBottom: 16 }}>
        <Form.Item name={[prefix, 'target_type']} label="类型" initialValue="llm_config" rules={[{ required: true }]}>
          <Select
            options={[
              { label: '模型配置', value: 'llm_config' },
              { label: 'Chat 接口', value: 'endpoint' },
            ]}
          />
        </Form.Item>
        <Form.Item name={[prefix, 'name']} label="展示名称" rules={[{ required: true, message: '请填写展示名称' }]}>
          <Input placeholder={targetType === 'llm_config' ? '例如：GPT-4.1 基线' : '例如：RAG 新版本'} />
        </Form.Item>
        {targetType === 'llm_config' ? (
          <Form.Item
            name={[prefix, 'llm_config_id']}
            label="LLM 配置"
            rules={[{ required: true, message: '请选择模型配置' }]}
          >
            <Select
              showSearch
              optionFilterProp="label"
              options={llmConfigs.map((item) => ({
                label: `${item.name} / ${item.model_name}`,
                value: item.id,
              }))}
            />
          </Form.Item>
        ) : (
          <>
            <Form.Item
              name={[prefix, 'endpoint_url']}
              label="接口地址"
              rules={[{ required: true, message: '请填写接口地址' }]}
            >
              <Input placeholder="https://example.com/chat" />
            </Form.Item>
            <Form.Item name={[prefix, 'transport_mode']} label="返回方式" initialValue="json">
              <Select
                options={[
                  { label: '普通 JSON', value: 'json' },
                  { label: 'SSE 流式', value: 'sse' },
                ]}
              />
            </Form.Item>
            <Form.Item name={[prefix, 'authorization']} label="Authorization">
              <Input placeholder="Bearer ... / AT-..." />
            </Form.Item>
            <Form.Item name={[prefix, 'extra_headers']} label="附加 Headers(JSON)" initialValue="{}">
              <TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
            </Form.Item>
            <Form.Item
              name={[prefix, 'request_body_template']}
              label="请求体模板(JSON)"
              initialValue={DEFAULT_ENDPOINT_BODY}
              rules={[{ required: true, message: '请填写请求体模板' }]}
            >
              <TextArea autoSize={{ minRows: 5, maxRows: 10 }} />
            </Form.Item>
            <Card size="small" type="inner" title="接口连通性测试">
              <Space direction="vertical" style={{ width: '100%' }} size={10}>
                <Input.TextArea
                  value={targetTestQuestion[prefix]}
                  onChange={(e) => setTargetTestQuestion((prev) => ({ ...prev, [prefix]: e.target.value }))}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  placeholder="输入一条测试问题"
                />
                <Button
                  icon={<ExperimentOutlined />}
                  loading={testingTarget === prefix}
                  onClick={() => handleTestTarget(prefix)}
                >
                  测试接口
                </Button>
              </Space>
            </Card>
          </>
        )}
      </Card>
    );
  };

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      if (!datasetHasUserInput) {
        message.error('盲测数据集必须包含 user_input 字段');
        return;
      }
      setSubmitting(true);
      await api.createBlindTest(values);
      message.success('人工盲测任务已创建，正在后台生成 A/B 回答');
      setCreateOpen(false);
      form.resetFields();
      fetchBaseData();
    } catch (error) {
      if (error) {
        message.error('创建盲测任务失败');
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleVote = async (vote: 'left' | 'right' | 'tie' | 'both_bad') => {
    if (!detailTask || !judgeRow) return;
    try {
      await api.voteBlindTestRow(detailTask.id, judgeRow.id, { vote, vote_note: voteNote || undefined });
      message.success('已记录人工判断');
      setJudgeRow(null);
      setVoteNote('');
      loadTaskDetail(detailTask.id, detailPage, detailPageSize, detailStatus);
    } catch {
      message.error('提交投票失败');
    }
  };

  const taskColumns = [
    { title: '任务名称', dataIndex: 'name', key: 'name', width: 200 },
    {
      title: '数据集',
      key: 'dataset',
      render: (_: unknown, record: BlindTestTask) => record.dataset?.name || datasets.find((item) => item.id === record.dataset_id)?.name || '-',
    },
    {
      title: '对比对象',
      key: 'targets',
      render: (_: unknown, record: BlindTestTask) => (
        <Space direction="vertical" size={0}>
          <Text>A：{record.target_a?.name || '-'}</Text>
          <Text>B：{record.target_b?.name || '-'}</Text>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (value: string) => <Tag color={statusColorMap[value] || 'default'}>{value}</Tag>,
    },
    {
      title: '进度',
      key: 'progress',
      width: 160,
      render: (_: unknown, record: BlindTestTask) => (
        <Space direction="vertical" size={0} style={{ width: '100%' }}>
          <Text type="secondary">{record.completed_rows}/{record.total_rows || 0}</Text>
          <div style={{ width: 120, height: 8, background: '#f0f0f0', borderRadius: 999 }}>
            <div
              style={{
                width: `${Math.round((record.progress || 0) * 100)}%`,
                height: '100%',
                background: record.status === 'completed' ? '#52c41a' : '#1677ff',
                borderRadius: 999,
              }}
            />
          </div>
        </Space>
      ),
    },
    {
      title: '已投票',
      key: 'voted_rows',
      width: 90,
      render: (_: unknown, record: BlindTestTask) => `${record.voted_rows}/${record.total_rows || 0}`,
    },
    {
      title: '操作',
      key: 'actions',
      width: 180,
      render: (_: unknown, record: BlindTestTask) => (
        <Space>
          <Button type="link" onClick={() => {
            setDetailTask(record);
            setDetailPage(1);
            setDetailStatus(undefined);
          }}>
            查看
          </Button>
          {['pending', 'running'].includes(record.status) && (
            <Button
              type="link"
              danger
              onClick={async () => {
                await api.cancelBlindTest(record.id);
                message.success('已取消盲测任务');
                fetchBaseData();
                if (detailTask?.id === record.id) {
                  loadTaskDetail(record.id);
                }
              }}
            >
              取消
            </Button>
          )}
        </Space>
      ),
    },
  ];

  const rowColumns = [
    { title: '#', dataIndex: 'row_index', key: 'row_index', width: 60 },
    {
      title: '问题',
      key: 'question',
      render: (_: unknown, record: BlindTestRowResult) => {
        const userInput = record.dataset_row?.data?.user_input;
        const preview = typeof userInput === 'string' ? userInput : JSON.stringify(userInput, null, 2);
        return (
          <Paragraph ellipsis={{ rows: 2, expandable: true, symbol: '展开' }} style={{ marginBottom: 0, maxWidth: 480 }}>
            {preview}
          </Paragraph>
        );
      },
    },
    {
      title: '答案状态',
      key: 'answerStatus',
      width: 150,
      render: (_: unknown, record: BlindTestRowResult) => {
        const okA = !!record.answer_a;
        const okB = !!record.answer_b;
        if (okA && okB) return <Tag color="green">已就绪</Tag>;
        if (record.answer_a_error || record.answer_b_error) return <Tag color="warning">部分失败</Tag>;
        return <Tag color="blue">生成中</Tag>;
      },
    },
    {
      title: '人工判断',
      key: 'vote',
      width: 130,
      render: (_: unknown, record: BlindTestRowResult) => {
        const voteLabel: Record<string, string> = {
          left: '左侧更好',
          right: '右侧更好',
          tie: '都好',
          both_bad: '都不好',
          skip: '跳过',
        };
        return record.vote ? <Tag color="purple">{voteLabel[record.vote] || record.vote}</Tag> : <Text type="secondary">待判断</Text>;
      },
    },
    {
      title: '去盲结果',
      key: 'winner',
      width: 220,
      render: (_: unknown, record: BlindTestRowResult) => {
        if (!record.vote) return <Text type="secondary">-</Text>;
        const leftSlot = (record.display_order?.[0] || 'a').toUpperCase();
        const rightSlot = (record.display_order?.[1] || (leftSlot === 'A' ? 'B' : 'A')).toUpperCase();
        if (record.vote === 'tie') {
          return (
            <Space direction="vertical" size={0}>
              <Tag color="success">A / B 都好</Tag>
              <Text type="secondary" style={{ fontSize: 12 }}>本条左侧={leftSlot}，右侧={rightSlot}</Text>
            </Space>
          );
        }
        if (record.vote === 'both_bad') {
          return (
            <Space direction="vertical" size={0}>
              <Tag color="error">A / B 都不好</Tag>
              <Text type="secondary" style={{ fontSize: 12 }}>本条左侧={leftSlot}，右侧={rightSlot}</Text>
            </Space>
          );
        }
        const leftSlotRaw = record.display_order?.[0] || 'a';
        const winner = record.vote === 'left'
          ? leftSlotRaw
          : (leftSlotRaw === 'a' ? 'b' : 'a');
        return (
          <Space direction="vertical" size={0}>
            <Tag color={winner === 'a' ? 'blue' : 'geekblue'}>{winner.toUpperCase()} 胜</Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>本条左侧={leftSlot}，右侧={rightSlot}</Text>
          </Space>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: BlindTestRowResult) => (
        <Button
          type="link"
          disabled={!record.answer_a && !record.answer_b}
          onClick={() => {
            setJudgeRow(record);
            setVoteNote(record.vote_note || '');
          }}
        >
          {record.vote ? '重判' : '评判'}
        </Button>
      ),
    },
  ];

  const renderDisplayedAnswer = (row: BlindTestRowResult, side: 'left' | 'right') => {
    const slot = row.display_order?.[side === 'left' ? 0 : 1] || 'a';
    const text = slot === 'a' ? row.answer_a : row.answer_b;
    const error = slot === 'a' ? row.answer_a_error : row.answer_b_error;
    return (
      <Card
        size="small"
        title={side === 'left' ? '回答甲' : '回答乙'}
        extra={error ? <Tag color="warning">生成失败</Tag> : <Tag color="blue">盲测中</Tag>}
      >
        {error ? <Alert type="warning" showIcon message={error} /> : <Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>{text || '暂无回答'}</Paragraph>}
      </Card>
    );
  };

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>人工盲测</Title>
          <Text type="secondary">同一批样本下比较两个应答端，既支持模型对比，也支持 Chat 接口 A/B。</Text>
        </div>
        <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => setCreateOpen(true)}>
          创建盲测任务
        </Button>
      </Space>

      <Alert
        style={{ marginBottom: 16 }}
        type="info"
        showIcon
        message="推荐用法"
        description="先用离线评测筛出关键数据集，再用人工盲测比较两个模型或新旧接口。自动评测看分数，人工盲测看体感和业务可接受度，两条线一起用最稳。"
      />

      <Table
        loading={loading}
        dataSource={tasks}
        columns={taskColumns}
        rowKey="id"
        pagination={false}
      />

      <Modal
        open={createOpen}
        title="创建人工盲测任务"
        width={920}
        onCancel={() => setCreateOpen(false)}
        onOk={handleCreate}
        confirmLoading={submitting}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            target_a: {
              target_type: 'llm_config',
              endpoint_url: DEFAULT_BLIND_TEST_ENDPOINT_URL,
              transport_mode: 'sse',
              authorization: DEFAULT_BLIND_TEST_AUTHORIZATION,
              extra_headers: DEFAULT_BLIND_TEST_HEADERS,
              request_body_template: DEFAULT_BLIND_TEST_BODY,
            },
            target_b: {
              target_type: 'llm_config',
              endpoint_url: DEFAULT_BLIND_TEST_ENDPOINT_URL,
              transport_mode: 'sse',
              authorization: DEFAULT_BLIND_TEST_AUTHORIZATION,
              extra_headers: DEFAULT_BLIND_TEST_HEADERS,
              request_body_template: DEFAULT_BLIND_TEST_BODY,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="任务名称" rules={[{ required: true, message: '请填写任务名称' }]}>
                <Input placeholder="例如：RAG 新旧版本主观对比" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="dataset_id" label="数据集" rules={[{ required: true, message: '请选择数据集' }]}>
                <Select
                  showSearch
                  optionFilterProp="label"
                  options={datasets.map((item) => ({
                    label: `${item.name} (${item.sample_type}, ${item.row_count} 条)`,
                    value: item.id,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={4}>
              <Form.Item name="sample_limit" label="抽样条数">
                <InputNumber min={1} max={500} style={{ width: '100%' }} placeholder="留空=全量" />
              </Form.Item>
            </Col>
          </Row>

          {selectedDataset && !datasetHasUserInput && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message="当前数据集缺少 user_input 字段"
              description="盲测要基于问题/对话去分别请求 A/B 两个应答端，所以数据集必须包含 user_input。"
            />
          )}

          <Row gutter={16}>
            <Col span={12}>{renderTargetFields('target_a', '对比对象 A')}</Col>
            <Col span={12}>{renderTargetFields('target_b', '对比对象 B')}</Col>
          </Row>
        </Form>
      </Modal>

      <Drawer
        open={!!detailTask}
        title={detailTask ? `盲测任务：${detailTask.name}` : '盲测任务'}
        width={1080}
        onClose={() => setDetailTask(null)}
        destroyOnClose
      >
        {detailTask && (
          <>
            <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="状态">
                <Tag color={statusColorMap[detailTask.status] || 'default'}>{detailTask.status}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="数据集">{detailTask.dataset?.name || '-'}</Descriptions.Item>
              <Descriptions.Item label="对比对象 A">{detailTask.target_a?.name || '-'}</Descriptions.Item>
              <Descriptions.Item label="对比对象 B">{detailTask.target_b?.name || '-'}</Descriptions.Item>
              <Descriptions.Item label="已生成">{detailTask.completed_rows}/{detailTask.total_rows || 0}</Descriptions.Item>
              <Descriptions.Item label="已投票">{detailTask.voted_rows}/{detailTask.total_rows || 0}</Descriptions.Item>
            </Descriptions>

            {detailSummary && (
              <>
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                  message="这里的 A/B 胜出是去盲后的真实归属"
                  description="人工判断时你看到的是“左侧 / 右侧”，但每条样本会随机交换展示顺序。所以左侧更好、右侧更好只是当时的位置；汇总里的 A 胜 / B 胜 是去盲后还原出来的真实结果。"
                />
                <Row gutter={16} style={{ marginBottom: 16 }}>
                <Col span={4}><Card><Statistic title="总样本" value={detailSummary.total_count} /></Card></Col>
                <Col span={4}><Card><Statistic title="A 胜" value={detailSummary.model_a_wins} prefix={<TrophyOutlined />} /></Card></Col>
                <Col span={4}><Card><Statistic title="B 胜" value={detailSummary.model_b_wins} prefix={<TrophyOutlined />} /></Card></Col>
                <Col span={4}><Card><Statistic title="都好" value={detailSummary.ties} prefix={<CheckCircleOutlined />} /></Card></Col>
                <Col span={4}><Card><Statistic title="都不好" value={detailSummary.both_bad} prefix={<CloseCircleOutlined />} /></Card></Col>
                <Col span={4}><Card><Statistic title="待投票" value={detailSummary.pending_vote_count} prefix={<MinusCircleOutlined />} /></Card></Col>
                </Row>
              </>
            )}

            {detailTask.error_message && (
              <Alert type="warning" showIcon style={{ marginBottom: 16 }} message={detailTask.error_message} />
            )}

            <Space style={{ marginBottom: 12 }}>
              <Select
                allowClear
                placeholder="筛选状态"
                value={detailStatus}
                style={{ width: 180 }}
                onChange={(value) => {
                  setDetailStatus(value);
                  setDetailPage(1);
                }}
                options={[
                  { label: '只看待投票', value: 'pending_vote' },
                  { label: '只看已投票', value: 'voted' },
                  { label: '只看已生成', value: 'ready' },
                ]}
              />
            </Space>

            <Table
              loading={detailLoading}
              rowKey="id"
              dataSource={detailRows}
              columns={rowColumns}
              pagination={{
                current: detailPage,
                pageSize: detailPageSize,
                total: detailRowsTotal,
                onChange: (nextPage, nextPageSize) => {
                  setDetailPage(nextPage);
                  setDetailPageSize(nextPageSize);
                },
              }}
            />
          </>
        )}
      </Drawer>

      <Modal
        open={!!judgeRow}
        title={judgeRow ? `样本 #${judgeRow.row_index}` : '人工评判'}
        width={1000}
        onCancel={() => setJudgeRow(null)}
        footer={null}
        destroyOnClose
      >
        {judgeRow && (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Card size="small" title="测试输入">
              <Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                {typeof judgeRow.dataset_row?.data?.user_input === 'string'
                  ? judgeRow.dataset_row?.data?.user_input
                  : JSON.stringify(judgeRow.dataset_row?.data?.user_input, null, 2)}
              </Paragraph>
            </Card>
            <Alert
              type="info"
              showIcon
              message="当前是盲测视角"
              description="这里的“左侧 / 右侧”只是展示位置，不代表 A / B。提交后系统会根据本条样本的随机顺序还原真实的 A/B 胜出结果。"
            />
            <Row gutter={16}>
              <Col span={12}>{renderDisplayedAnswer(judgeRow, 'left')}</Col>
              <Col span={12}>{renderDisplayedAnswer(judgeRow, 'right')}</Col>
            </Row>
            <TextArea
              value={voteNote}
              onChange={(e) => setVoteNote(e.target.value)}
              placeholder="可选：记录你为什么这么判断，例如“左侧更完整，但略啰嗦”"
              autoSize={{ minRows: 3, maxRows: 6 }}
            />
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
              <Space size={20} wrap>
                <Button type="primary" size="large" onClick={() => handleVote('left')}>左侧更好</Button>
                <Button type="primary" size="large" onClick={() => handleVote('right')}>右侧更好</Button>
                <Button size="large" onClick={() => handleVote('tie')}>都好</Button>
                <Button danger size="large" onClick={() => handleVote('both_bad')}>都不好</Button>
              </Space>
            </div>
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default BlindTestPage;

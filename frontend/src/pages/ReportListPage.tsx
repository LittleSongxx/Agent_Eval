import React, { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Input,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { FileTextOutlined, QuestionCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import type { ReportListItem, ReportListResponse } from '../types';
import * as api from '../services/api';

const { Title, Text } = Typography;

const HelpTitle: React.FC<{ label: string; tip: string }> = ({ label, tip }) => (
  <Space size={4}>
    <span>{label}</span>
    <Tooltip title={tip}>
      <QuestionCircleOutlined style={{ color: '#8c8c8c', fontSize: 12 }} />
    </Tooltip>
  </Space>
);

const statusColorMap: Record<string, string> = {
  pending: 'blue',
  running: 'orange',
  completed: 'green',
  failed: 'red',
  cancelled: 'default',
};

const statusLabelMap: Record<string, string> = {
  pending: '等待中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const ReportListPage: React.FC = () => {
  const navigate = useNavigate();
  const [items, setItems] = useState<ReportListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [status, setStatus] = useState<string | undefined>();
  const [evaluationMode, setEvaluationMode] = useState<string | undefined>();
  const [keyword, setKeyword] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchReports = async (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true);
    try {
      const data: ReportListResponse = await api.listReports({
        page: nextPage,
        page_size: nextPageSize,
        status,
        evaluation_mode: evaluationMode,
        keyword: keyword || undefined,
      });
      setItems(data.items || []);
      setTotal(data.total || 0);
      setPage(data.page || nextPage);
      setPageSize(data.page_size || nextPageSize);
    } catch {
      message.error('加载评测报告失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports(1, pageSize);
  }, [status, evaluationMode]);

  const columns = [
    {
      title: '报告任务',
      key: 'task',
      width: 260,
      render: (_: unknown, record: ReportListItem) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.task_name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>#{record.eval_id} · {new Date(record.created_at).toLocaleString('zh-CN')}</Text>
        </Space>
      ),
    },
    {
      title: '数据集 / 场景',
      key: 'dataset',
      width: 260,
      render: (_: unknown, record: ReportListItem) => (
        <Space direction="vertical" size={0}>
          <Text>{record.dataset_name || '-'}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.scenario_name || '-'}</Text>
        </Space>
      ),
    },
    {
      title: '评测模式',
      key: 'mode',
      width: 150,
      render: (_: unknown, record: ReportListItem) => (
        <Space direction="vertical" size={0}>
          <Tag color={record.evaluation_mode === 'endpoint' ? 'purple' : 'default'}>
            {record.evaluation_mode === 'endpoint' ? '接口实时评测' : '已有结果评测'}
          </Tag>
          {record.endpoint_name && <Text type="secondary" style={{ fontSize: 12 }}>{record.endpoint_name}</Text>}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (value: string) => <Tag color={statusColorMap[value] || 'default'}>{statusLabelMap[value] || value}</Tag>,
    },
    {
      title: <HelpTitle label="样本通过率" tip="按样本维度统计：样本通过数 / 已评测样本数。它不是单个指标的通过率。" />,
      key: 'pass_rate',
      width: 180,
      render: (_: unknown, record: ReportListItem) => (
        <Space direction="vertical" size={2} style={{ width: '100%' }}>
          <Progress
            percent={Math.round((record.pass_rate || 0) * 100)}
            size="small"
            status={record.status === 'failed' ? 'exception' : record.status === 'completed' ? 'success' : 'active'}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            通过 {record.pass_count} / 失败 {record.fail_count} / 异常 {record.error_count}
          </Text>
        </Space>
      ),
    },
    {
      title: '进度',
      key: 'progress',
      width: 130,
      render: (_: unknown, record: ReportListItem) => `${record.completed_rows || 0}/${record.total_rows || record.total_count || 0}`,
    },
    {
      title: '完成时间',
      dataIndex: 'finished_at',
      key: 'finished_at',
      width: 180,
      render: (value?: string | null) => value ? new Date(value).toLocaleString('zh-CN') : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_: unknown, record: ReportListItem) => (
        <Button
          type="link"
          icon={<FileTextOutlined />}
          disabled={record.status !== 'completed'}
          onClick={() => navigate(`/reports/${record.eval_id}`)}
        >
          查看报告
        </Button>
      ),
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <Title level={4} style={{ marginBottom: 4 }}>评测报告</Title>
          <Text type="secondary">集中查看评测任务的报告摘要、样本通过率、失败数和接口实时评测结果。</Text>
        </div>
        <Button icon={<ReloadOutlined />} onClick={() => fetchReports(page, pageSize)}>刷新</Button>
      </Space>

      <Card style={{ marginBottom: 16 }}>
        <Space wrap>
          <Input.Search
            placeholder="搜索任务名称"
            allowClear
            style={{ width: 260 }}
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onSearch={() => fetchReports(1, pageSize)}
          />
          <Select
            allowClear
            placeholder="状态"
            style={{ width: 160 }}
            value={status}
            onChange={(value) => setStatus(value)}
            options={[
              { label: '已完成', value: 'completed' },
              { label: '运行中', value: 'running' },
              { label: '等待中', value: 'pending' },
              { label: '失败', value: 'failed' },
              { label: '已取消', value: 'cancelled' },
            ]}
          />
          <Select
            allowClear
            placeholder="评测模式"
            style={{ width: 180 }}
            value={evaluationMode}
            onChange={(value) => setEvaluationMode(value)}
            options={[
              { label: '已有结果评测', value: 'offline' },
              { label: '接口实时评测', value: 'endpoint' },
            ]}
          />
        </Space>
      </Card>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="只有已完成任务可查看详情报告"
        description="运行中任务会先显示进度和当前统计，完成后可以进入报告详情做样本分析、人工复核和基线对比。"
      />

      <Card>
        <Table
          rowKey="eval_id"
          loading={loading}
          columns={columns}
          dataSource={items}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            showTotal: (count) => `共 ${count} 份报告`,
            onChange: (nextPage, nextPageSize) => fetchReports(nextPage, nextPageSize),
          }}
        />
      </Card>
    </div>
  );
};

export default ReportListPage;

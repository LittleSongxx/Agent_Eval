import React, { useEffect, useState, useCallback } from 'react';
import {
  Tabs,
  Table,
  Card,
  Row,
  Col,
  Statistic,
  Tag,
  Button,
  Space,
  Select,
  Descriptions,
  Drawer,
  List,
  Typography,
  Spin,
  message,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate } from 'react-router-dom';
import type { ReportSummary, EvalRowResult, PaginatedResponse } from '../types';
import * as api from '../services/api';

const { Title, Text, Paragraph } = Typography;

const ReportDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const evalId = Number(id);

  const [summary, setSummary] = useState<ReportSummary | null>(null);
  const [rows, setRows] = useState<EvalRowResult[]>([]);
  const [rowsTotal, setRowsTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedRow, setSelectedRow] = useState<EvalRowResult | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getReportSummary(evalId);
      setSummary(data);
    } catch {
      message.error('加载报告摘要失败');
    } finally {
      setLoading(false);
    }
  }, [evalId]);

  const fetchRows = useCallback(async () => {
    setRowsLoading(true);
    try {
      const data: PaginatedResponse<EvalRowResult> = await api.getReportRows(
        evalId,
        page,
        pageSize,
        statusFilter
      );
      setRows(data.items || []);
      setRowsTotal(data.total || 0);
    } catch {
      message.error('加载评测明细失败');
    } finally {
      setRowsLoading(false);
    }
  }, [evalId, page, pageSize, statusFilter]);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  useEffect(() => {
    fetchRows();
  }, [fetchRows]);

  const openDetail = async (row: EvalRowResult) => {
    setDrawerOpen(true);
    setDetailLoading(true);
    try {
      const detail = await api.getReportRowDetail(evalId, row.id);
      setSelectedRow(detail);
    } catch {
      message.error('加载详情失败');
      setSelectedRow(row);
    } finally {
      setDetailLoading(false);
    }
  };

  const passRateColor = (rate: number) => {
    if (rate >= 0.8) return '#52c41a';
    if (rate >= 0.6) return '#faad14';
    return '#ff4d4f';
  };

  const metricNames = summary ? Object.keys(summary.metric_summary || {}) : [];

  const detailColumns = [
    {
      title: '#',
      dataIndex: 'row_index',
      key: 'row_index',
      width: 60,
    },
    {
      title: '状态',
      key: 'status',
      width: 80,
      render: (_: unknown, record: EvalRowResult) => {
        if (record.error) {
          return <ExclamationCircleOutlined style={{ color: '#faad14', fontSize: 18 }} />;
        }
        if (record.is_pass === true) {
          return <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 18 }} />;
        }
        if (record.is_pass === false) {
          return <CloseCircleOutlined style={{ color: '#ff4d4f', fontSize: 18 }} />;
        }
        return <Tag>-</Tag>;
      },
    },
    ...metricNames.map((metric) => ({
      title: metric,
      key: metric,
      width: 100,
      render: (_: unknown, record: EvalRowResult) => {
        const ms = record.metric_scores?.[metric];
        if (!ms) return '-';
        const score = ms.score;
        if (score === null || score === undefined) return '-';
        return (
          <span style={{ color: score >= 0.6 ? '#52c41a' : '#ff4d4f' }}>
            {(score * 100).toFixed(1)}%
          </span>
        );
      },
    })),
    {
      title: '耗时',
      dataIndex: 'execution_time_ms',
      key: 'execution_time_ms',
      width: 80,
      render: (v: number | null) => (v !== null && v !== undefined ? `${v}ms` : '-'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (_: unknown, record: EvalRowResult) => (
        <Button type="link" size="small" onClick={() => openDetail(record)}>
          详情
        </Button>
      ),
    },
  ];

  const evalTask = summary?.eval_task;

  const renderOverview = () => {
    if (!summary) return null;

    const metricSummaryData = Object.entries(summary.metric_summary || {}).map(
      ([name, stats]) => ({
        key: name,
        metric: name,
        mean: stats.mean,
        min: stats.min,
        max: stats.max,
        pass_rate: stats.pass_rate,
      })
    );

    const duration =
      evalTask?.started_at && evalTask?.finished_at
        ? Math.round(
            (new Date(evalTask.finished_at).getTime() -
              new Date(evalTask.started_at).getTime()) /
              1000
          )
        : null;

    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="通过率"
                value={(summary.pass_rate * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: passRateColor(summary.pass_rate) }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="总条数" value={summary.total_count} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="通过"
                value={summary.pass_count}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="不通过"
                value={summary.fail_count}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>

        <Card title="各指标得分概览" style={{ marginBottom: 24 }}>
          <Table
            rowKey="metric"
            dataSource={metricSummaryData}
            pagination={false}
            size="small"
            columns={[
              { title: '指标名称', dataIndex: 'metric', key: 'metric' },
              {
                title: '平均得分',
                dataIndex: 'mean',
                key: 'mean',
                render: (v: number) => (v * 100).toFixed(1) + '%',
              },
              {
                title: '最低分',
                dataIndex: 'min',
                key: 'min',
                render: (v: number) => (v * 100).toFixed(1) + '%',
              },
              {
                title: '最高分',
                dataIndex: 'max',
                key: 'max',
                render: (v: number) => (v * 100).toFixed(1) + '%',
              },
              {
                title: '通过率',
                dataIndex: 'pass_rate',
                key: 'pass_rate',
                render: (v: number) => (
                  <span style={{ color: passRateColor(v) }}>
                    {(v * 100).toFixed(1)}%
                  </span>
                ),
              },
            ]}
          />
        </Card>

        <Card title="基本信息">
          <Descriptions column={2} size="small">
            <Descriptions.Item label="评测名称">
              {evalTask?.name || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="数据集">
              {evalTask?.dataset?.name || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="场景">
              {evalTask?.scenario?.name || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="LLM">
              {evalTask?.llm_config?.name || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="执行时间">
              {evalTask?.started_at
                ? new Date(evalTask.started_at).toLocaleString('zh-CN')
                : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="耗时">
              {duration !== null ? `${duration} 秒` : '-'}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </div>
    );
  };

  const renderDetail = () => (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Text>状态筛选:</Text>
        <Select
          style={{ width: 140 }}
          allowClear
          placeholder="全部"
          value={statusFilter}
          onChange={(val) => {
            setStatusFilter(val);
            setPage(1);
          }}
          options={[
            { label: '通过', value: 'pass' },
            { label: '不通过', value: 'fail' },
            { label: '错误', value: 'error' },
          ]}
        />
      </Space>

      <Table
        rowKey="id"
        loading={rowsLoading}
        columns={detailColumns}
        dataSource={rows}
        scroll={{ x: 'max-content' }}
        pagination={{
          current: page,
          pageSize: pageSize,
          total: rowsTotal,
          showSizeChanger: true,
          showTotal: (t) => `共 ${t} 条`,
          onChange: (p, ps) => {
            setPage(p);
            setPageSize(ps);
          },
        }}
      />

      <Drawer
        title={`第 ${selectedRow?.row_index ?? '-'} 条评测详情`}
        open={drawerOpen}
        onClose={() => {
          setDrawerOpen(false);
          setSelectedRow(null);
        }}
        width={640}
      >
        <Spin spinning={detailLoading}>
          {selectedRow && (
            <>
              <Title level={5}>原始数据</Title>
              {selectedRow.dataset_row?.data ? (
                <Descriptions
                  column={1}
                  bordered
                  size="small"
                  style={{ marginBottom: 24 }}
                >
                  {Object.entries(selectedRow.dataset_row.data).map(([key, value]) => (
                    <Descriptions.Item key={key} label={key}>
                      <Text
                        style={{
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-all',
                        }}
                      >
                        {typeof value === 'string'
                          ? value
                          : JSON.stringify(value, null, 2)}
                      </Text>
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              ) : (
                <Text type="secondary">无原始数据</Text>
              )}

              <Title level={5} style={{ marginTop: 24 }}>
                指标评分
              </Title>
              <List
                dataSource={Object.entries(selectedRow.metric_scores || {})}
                renderItem={([metric, result]) => (
                  <List.Item>
                    <List.Item.Meta
                      title={
                        <Space>
                          <Text strong>{metric}</Text>
                          {result.score !== null && result.score !== undefined ? (
                            <Tag color={result.score >= 0.6 ? 'green' : 'red'}>
                              {(result.score * 100).toFixed(1)}%
                            </Tag>
                          ) : (
                            <Tag>N/A</Tag>
                          )}
                        </Space>
                      }
                      description={
                        <Paragraph
                          type="secondary"
                          style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}
                        >
                          {result.reason || '无评分理由'}
                        </Paragraph>
                      }
                    />
                  </List.Item>
                )}
              />

              {selectedRow.error && (
                <>
                  <Title level={5} style={{ marginTop: 24, color: '#ff4d4f' }}>
                    错误信息
                  </Title>
                  <Paragraph type="danger" style={{ whiteSpace: 'pre-wrap' }}>
                    {selectedRow.error}
                  </Paragraph>
                </>
              )}

              {selectedRow.execution_time_ms !== null && (
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">
                    执行耗时: {selectedRow.execution_time_ms}ms
                  </Text>
                </div>
              )}
            </>
          )}
        </Spin>
      </Drawer>
    </div>
  );

  return (
    <Spin spinning={loading}>
      <Space style={{ marginBottom: 16 }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/evaluations')}
        >
          返回评测列表
        </Button>
        <Title level={4} style={{ margin: 0 }}>
          评测报告
        </Title>
      </Space>

      <Tabs
        defaultActiveKey="overview"
        items={[
          {
            key: 'overview',
            label: '评测总览',
            children: renderOverview(),
          },
          {
            key: 'detail',
            label: '逐条明细',
            children: renderDetail(),
          },
        ]}
      />
    </Spin>
  );
};

export default ReportDetailPage;

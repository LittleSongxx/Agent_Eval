import React, { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Space,
  message,
  Typography,
  Card,
  Statistic,
  Row,
  Col,
  Select,
  Descriptions,
  Tag,
} from 'antd';
import {
  DownloadOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import type { EvalTask, ReportSummary, EvalRowResult } from '../types';
import * as api from '../services/api';

const { Title, Text } = Typography;

const ReportPage: React.FC = () => {
  const [tasks, setTasks] = useState<EvalTask[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null);
  const [summary, setSummary] = useState<ReportSummary | null>(null);
  const [rows, setRows] = useState<EvalRowResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [rowsLoading, setRowsLoading] = useState(false);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const res: EvalTask[] = await api.listEvaluations();
        const completedTasks = res.filter((t: EvalTask) => t.status === 'completed');
        setTasks(completedTasks);
      } catch {
        message.error('加载任务列表失败');
      }
    };
    fetchTasks();
  }, []);

  const handleSelectTask = async (taskId: number) => {
    setSelectedTaskId(taskId);
    setLoading(true);
    setRowsLoading(true);
    try {
      const [summaryData, rowsData] = await Promise.all([
        api.getReportSummary(taskId),
        api.getReportRows(taskId, 1, 50),
      ]);
      setSummary(summaryData);
      setRows(rowsData.items);
    } catch {
      message.error('加载报告失败');
    } finally {
      setLoading(false);
      setRowsLoading(false);
    }
  };

  const handleExport = async (format: 'csv' | 'json') => {
    if (!selectedTaskId) return;
    try {
      const blob = format === 'json'
        ? new Blob([JSON.stringify({ summary, rows }, null, 2)], { type: 'application/json' })
        : new Blob([toCsv(rows, metricKeys)], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${selectedTaskId}.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);
      message.success('导出成功');
    } catch {
      message.error('导出失败');
    }
  };

  const metricKeys = summary
    ? Object.keys(summary.metric_summary ?? {})
    : [];

  const toCsv = (items: EvalRowResult[], metrics: string[]) => {
    const headers = ['row_index', 'status', ...metrics, 'execution_time_ms', 'error'];
    const escapeCell = (value: unknown) => {
      const text = value == null ? '' : String(value);
      return `"${text.replace(/"/g, '""')}"`;
    };
    const lines = items.map((item) => {
      const metricValues = metrics.map((key) => item.metric_scores?.[key]?.score ?? '');
      return [
        item.row_index,
        item.error ? 'error' : item.is_pass ? 'pass' : 'fail',
        ...metricValues,
        item.execution_time_ms ?? '',
        item.error ?? '',
      ].map(escapeCell).join(',');
    });
    return [headers.join(','), ...lines].join('\n');
  };

  const rowColumns = [
    {
      title: '行号',
      dataIndex: 'row_index',
      key: 'row_index',
      width: 70,
    },
    {
      title: '状态',
      key: 'status',
      width: 90,
      render: (_: unknown, record: EvalRowResult) => {
        if (record.error) return <Tag color="warning">异常</Tag>;
        return record.is_pass ? <Tag color="success">通过</Tag> : <Tag color="error">失败</Tag>;
      },
    },
    {
      title: '输入',
      key: 'input',
      ellipsis: true,
      render: (_: unknown, record: EvalRowResult) =>
        record.dataset_row?.data?.user_input ?? '-',
    },
    ...metricKeys.map((key) => ({
      title: key,
      key,
      render: (_: unknown, record: EvalRowResult) =>
        record.metric_scores?.[key]?.score?.toFixed(3) ?? '-',
      width: 100,
    })),
    {
      title: '延迟(ms)',
      dataIndex: 'execution_time_ms',
      key: 'execution_time_ms',
      width: 90,
      render: (v: number | null) => v?.toFixed(0) ?? '-',
    },
    {
      title: '错误',
      dataIndex: 'error',
      key: 'error',
      render: (v: string | null) => (v ? <Tag color="error">{v}</Tag> : '-'),
    },
  ];

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>评测报告</Title>
        <Space>
          <Select
            placeholder="选择已完成的评测任务"
            style={{ width: 300 }}
            value={selectedTaskId}
            onChange={handleSelectTask}
            options={tasks.map((t) => ({
              label: `任务 #${t.id} (场景#${t.scenario_id})`,
              value: t.id,
            }))}
          />
          {selectedTaskId && (
            <Space>
              <Button icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
                导出 CSV
              </Button>
              <Button icon={<DownloadOutlined />} onClick={() => handleExport('json')}>
                导出 JSON
              </Button>
            </Space>
          )}
        </Space>
      </div>

      {summary && (
        <>
          <Descriptions bordered size="small" column={3} style={{ marginBottom: 16 }}>
            <Descriptions.Item label="场景">{summary.eval_task.scenario?.name ?? `#${summary.eval_task.scenario_id}`}</Descriptions.Item>
            <Descriptions.Item label="数据集">{summary.eval_task.dataset?.name ?? `#${summary.eval_task.dataset_id}`}</Descriptions.Item>
            <Descriptions.Item label="完成时间">
              {summary.eval_task.finished_at ? new Date(summary.eval_task.finished_at).toLocaleString('zh-CN') : '-'}
            </Descriptions.Item>
          </Descriptions>

          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card>
                <Statistic title="总行数" value={summary.total_count} />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic title="通过" value={summary.pass_count} valueStyle={{ color: '#3f8600' }} />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic title="失败" value={summary.fail_count} valueStyle={{ color: '#cf1322' }} />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="通过率"
                  value={summary.pass_rate * 100}
                  precision={1}
                  suffix="%"
                />
              </Card>
            </Col>
          </Row>

          {metricKeys.length > 0 && (
            <Card
              title={<Space><BarChartOutlined /><Text strong>指标平均分</Text></Space>}
              size="small"
              style={{ marginBottom: 24 }}
            >
              <Row gutter={16}>
                {metricKeys.map((key) => (
                  <Col span={6} key={key}>
                    <Statistic
                      title={key}
                      value={summary.metric_summary[key]?.mean ?? 0}
                      precision={3}
                    />
                  </Col>
                ))}
              </Row>
            </Card>
          )}

          <Title level={5}>详细结果</Title>
          <Table
            rowKey="id"
            loading={rowsLoading}
            columns={rowColumns}
            dataSource={rows}
            size="small"
            scroll={{ x: 'max-content' }}
            pagination={{ pageSize: 20 }}
          />
        </>
      )}

      {!summary && !loading && (
        <div style={{ textAlign: 'center', padding: '80px 0', color: '#999' }}>
          <BarChartOutlined style={{ fontSize: 48, marginBottom: 16, display: 'block' }} />
          <Text type="secondary">请从上方选择一个已完成的评测任务以查看报告</Text>
        </div>
      )}
    </>
  );
};

export default ReportPage;

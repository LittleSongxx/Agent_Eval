import React, { useEffect, useState, useCallback } from 'react';
import {
  Tabs, Table, Card, Row, Col, Statistic, Tag, Button, Space, Select,
  Descriptions, Drawer, Typography, Spin, message, Progress, Divider, Tooltip, Alert,
} from 'antd';
import {
  CheckCircleOutlined, CloseCircleOutlined, ExclamationCircleOutlined,
  ArrowLeftOutlined, QuestionCircleOutlined, InfoCircleOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate } from 'react-router-dom';
import type { ReportSummary, EvalRowResult, PaginatedResponse } from '../types';
import * as api from '../services/api';
import { MetricHelpDrawer, MetricHelpIcon } from '../components/MetricHelpDrawer';
import { getMetricInfo, groupMetricNames, type MetricLayer } from '../utils/metricLayers';

const { Title, Text, Paragraph } = Typography;

const layerHeaderCell = (group: MetricLayer, level: 'group' | 'metric' = 'group') => ({
  style: {
    background: level === 'group' ? group.headerBg : group.subHeaderBg,
    borderBottom: `2px solid ${group.headerBorder}`,
    color: group.headerText,
    fontWeight: 600,
  },
});

const renderLayerHeaderTitle = (group: MetricLayer) => (
  <div style={{ lineHeight: 1.25 }}>
    <div>
      <Tag color={group.color} style={{ marginRight: 6 }}>{group.name}</Tag>
    </div>
    {group.description && (
      <div
        style={{
          marginTop: 3,
          color: group.headerText,
          fontSize: 11,
          fontWeight: 400,
          opacity: 0.82,
          whiteSpace: 'normal',
        }}
      >
        （{group.description}）
      </div>
    )}
  </div>
);

const formatScore = (score: any, metricName: string): { display: string; color: string; explain: string } => {
  if (score === null || score === undefined) return { display: 'N/A', color: '#999', explain: '指标计算出错，无法得出分数' };
  if (typeof score === 'string') {
    const pass = ['pass', 'yes', 'true', '1'].includes(score.toLowerCase());
    return { display: score, color: pass ? '#52c41a' : '#ff4d4f', explain: pass ? '通过' : '不通过' };
  }
  const num = Number(score);
  const isBinary = ['agent_goal_accuracy', 'harmfulness', 'coherence'].includes(metricName);
  if (isBinary) {
    return num >= 0.5
      ? { display: `${num}（是）`, color: '#52c41a', explain: '满足条件' }
      : { display: `${num}（否）`, color: '#ff4d4f', explain: '不满足条件' };
  }
  const pct = (num * 100).toFixed(1);
  let level = '优秀'; let color = '#52c41a';
  if (num < 0.5) { level = '较差'; color = '#ff4d4f'; }
  else if (num < 0.7) { level = '一般'; color = '#faad14'; }
  else if (num < 0.9) { level = '良好'; color = '#1677ff'; }
  return { display: `${pct}%`, color, explain: `${level}（${num.toFixed(4)} / 1.0）` };
};

const FIELD_META: Record<string, { label: string; group: string; tip: string }> = {
  user_input:           { label: '测试问题 / 对话记录', group: 'input',   tip: '你设计的测试用例输入' },
  response:             { label: '被测系统的回答',      group: 'output',  tip: '从你的 RAG/Agent 系统收集的回答，评测引擎对它打分' },
  retrieved_contexts:   { label: '被测系统检索到的上下文', group: 'output', tip: '从你的 RAG 检索模块收集的文档片段，忠实度指标会检查回答是否基于这些内容' },
  retrieved_context_ids:{ label: '被测系统召回的文档 ID', group: 'output', tip: '从你的 RAG 检索模块收集的文档唯一标识，用于 HitRate@K 和 MRR 等确定性检索指标' },
  reference:            { label: '标准答案（你写的）',    group: 'ref',    tip: '你人工标注的正确答案，评测 LLM 用它来衡量回答质量' },
  reference_tool_calls: { label: '期望的工具调用（你写的）', group: 'ref',  tip: '你标注的 Agent 应该调用哪些工具，工具调用准确度指标会拿实际调用与这里对比' },
  reference_topics:     { label: '允许的话题范围（你写的）', group: 'ref',  tip: '你标注的对话应围绕的话题，话题遵守度指标会检查 AI 是否跑题' },
  reference_contexts:   { label: '参考上下文（你写的）',    group: 'ref',   tip: '你标注的理想检索结果' },
  reference_context_ids:{ label: '期望召回的文档 ID（你写的）', group: 'ref', tip: '你标注的正确文档唯一标识，用于判断检索是否命中以及排序是否合理' },
};

const DataFieldsView: React.FC<{ data: Record<string, any> }> = ({ data }) => {
  const entries = Object.entries(data);
  const inputFields = entries.filter(([k]) => FIELD_META[k]?.group === 'input' || k === 'user_input');
  const outputFields = entries.filter(([k]) => FIELD_META[k]?.group === 'output' || ['response', 'retrieved_contexts'].includes(k));
  const refFields = entries.filter(([k]) => FIELD_META[k]?.group === 'ref' || k.startsWith('reference'));
  const classified = new Set([...inputFields, ...outputFields, ...refFields].map(([k]) => k));
  const otherFields = entries.filter(([k]) => !classified.has(k));

  const renderValue = (value: any) => {
    if (value === null || value === undefined) return <Text type="secondary">（空）</Text>;
    if (typeof value === 'object') {
      return (
        <pre style={{ margin: 0, fontSize: 12, maxHeight: 180, overflow: 'auto', background: '#f9f9f9', padding: 8, borderRadius: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    }
    return <Text style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{String(value)}</Text>;
  };

  const renderGroup = (title: string, desc: string, color: string, fields: [string, any][]) => {
    if (fields.length === 0) return null;
    return (
      <div style={{ marginBottom: 20 }}>
        <div style={{ marginBottom: 8 }}>
          <Tag color={color} style={{ fontSize: 12 }}>{title}</Tag>
          <Text type="secondary" style={{ fontSize: 12 }}>{desc}</Text>
        </div>
        {fields.map(([key, value]) => {
          const meta = FIELD_META[key];
          return (
            <div key={key} style={{ marginBottom: 12, paddingLeft: 4, borderLeft: '3px solid #f0f0f0', paddingBottom: 4 }}>
              <div style={{ marginBottom: 4 }}>
                <Text strong style={{ fontSize: 13 }}>{meta?.label || key}</Text>
                {meta?.tip && (
                  <Tooltip title={meta.tip}>
                    <QuestionCircleOutlined style={{ color: '#1677ff', marginLeft: 6, fontSize: 12 }} />
                  </Tooltip>
                )}
              </div>
              <div style={{ paddingLeft: 4 }}>{renderValue(value)}</div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div>
      <Alert
        message="离线评测模式下，以下所有字段都是你填入数据集的。区别在于数据的原始来源不同。"
        type="info" showIcon style={{ marginBottom: 16, fontSize: 12 }}
      />
      {renderGroup('📝 测试输入', '你设计的测试问题', 'blue', inputFields)}
      {renderGroup('🤖 被测系统输出', '从你的 AI 系统运行结果中收集，填入数据集。评测引擎对这些内容打分', 'purple', outputFields)}
      {renderGroup('✅ 人工标注的标准答案', '你写的正确答案，评测 LLM 拿 AI 输出与这里对比来打分', 'orange', refFields)}
      {otherFields.length > 0 && renderGroup('📎 其他', '', 'default', otherFields)}
    </div>
  );
};

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
  const [helpMetric, setHelpMetric] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    try { setSummary(await api.getReportSummary(evalId)); } catch { message.error('加载报告失败'); } finally { setLoading(false); }
  }, [evalId]);

  const fetchRows = useCallback(async () => {
    setRowsLoading(true);
    try {
      const data: PaginatedResponse<EvalRowResult> = await api.getReportRows(evalId, page, pageSize, statusFilter);
      setRows(data.items || []); setRowsTotal(data.total || 0);
    } catch { message.error('加载明细失败'); } finally { setRowsLoading(false); }
  }, [evalId, page, pageSize, statusFilter]);

  useEffect(() => { fetchSummary(); }, [fetchSummary]);
  useEffect(() => { fetchRows(); }, [fetchRows]);

  const openDetail = async (row: EvalRowResult) => {
    setDrawerOpen(true); setDetailLoading(true);
    try { setSelectedRow(await api.getReportRowDetail(evalId, row.id)); }
    catch { setSelectedRow(row); } finally { setDetailLoading(false); }
  };

  const passRateColor = (r: number) => r >= 0.8 ? '#52c41a' : r >= 0.6 ? '#faad14' : '#ff4d4f';
  const metricNames = summary ? Object.keys(summary.metric_summary || {}) : [];
  const evalTask = summary?.eval_task;
  const metricHelp = (metric: string) => (
    <MetricHelpIcon metricName={metric} onClick={() => setHelpMetric(metric)} />
  );

  const detailColumns = [
    { title: '#', dataIndex: 'row_index', key: 'row_index', width: 50 },
    {
      title: '状态', key: 'status', width: 70,
      render: (_: unknown, r: EvalRowResult) =>
        r.error ? <Tag color="warning">异常</Tag> :
        r.is_pass === true ? <Tag color="success">通过</Tag> :
        r.is_pass === false ? <Tag color="error">不通过</Tag> : <Tag>-</Tag>,
    },
    ...groupMetricNames(metricNames).map((group) => ({
      title: renderLayerHeaderTitle(group),
      key: group.key,
      onHeaderCell: () => layerHeaderCell(group, 'group'),
      children: group.metrics.map((metric) => ({
        title: (
          <Space size={4}>
            <span>{getMetricInfo(metric).shortName}</span>
            {metricHelp(metric)}
          </Space>
        ),
        key: metric,
        width: 130,
        onHeaderCell: () => layerHeaderCell(group, 'metric'),
        render: (_: unknown, record: EvalRowResult) => {
          const ms = record.metric_scores?.[metric];
          if (!ms) return <Text type="secondary">-</Text>;
          const { display, color } = formatScore(ms.score, metric);
          return <Text style={{ color, fontWeight: 500 }}>{display}</Text>;
        },
      })),
    })),
    { title: '耗时', dataIndex: 'execution_time_ms', key: 'time', width: 70, render: (v: number | null) => v != null ? `${v}ms` : '-' },
    {
      title: '操作', key: 'actions', width: 70,
      render: (_: unknown, r: EvalRowResult) => <Button type="link" size="small" onClick={() => openDetail(r)}>详情</Button>,
    },
  ];

  const renderOverview = () => {
    if (!summary) return null;
    const metricSummary = summary.metric_summary || {};
    const metricGroups = groupMetricNames(Object.keys(metricSummary));
    const duration = evalTask?.started_at && evalTask?.finished_at
      ? Math.round((new Date(evalTask.finished_at).getTime() - new Date(evalTask.started_at).getTime()) / 1000) : null;

    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}><Card><Statistic title="通过率" value={(summary.pass_rate * 100).toFixed(1)} suffix="%" valueStyle={{ color: passRateColor(summary.pass_rate) }} /></Card></Col>
          <Col span={6}><Card><Statistic title="总条数" value={summary.total_count} /></Card></Col>
          <Col span={6}><Card><Statistic title="通过" value={summary.pass_count} valueStyle={{ color: '#52c41a' }} /></Card></Col>
          <Col span={6}><Card><Statistic title="不通过" value={summary.fail_count} valueStyle={{ color: '#ff4d4f' }} /></Card></Col>
        </Row>

        <Card title="各指标得分概览" extra={<Text type="secondary">按 RAG 评测层级归类，分数范围 0~1，越高越好</Text>} style={{ marginBottom: 24 }}>
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {metricGroups.map((group) => {
              const dataSource = group.metrics.map((name) => ({
                key: name,
                metric: name,
                ...metricSummary[name],
              }));
              return (
                <Card
                  key={group.key}
                  size="small"
                  title={
                    <Space size={6} wrap>
                      <Tag color={group.color}>{group.name}</Tag>
                      <Text type="secondary" style={{ fontSize: 12 }}>（{group.description}）</Text>
                    </Space>
                  }
                >
                  <Table rowKey="metric" dataSource={dataSource} pagination={false} size="small" columns={[
                    {
                      title: '指标', dataIndex: 'metric', key: 'metric', width: 200,
                      onHeaderCell: () => layerHeaderCell(group, 'metric'),
                      render: (name: string) => (
                        <Space>
                          <Text strong>{getMetricInfo(name).shortName}</Text>
                          {metricHelp(name)}
                        </Space>
                      ),
                    },
                    { title: '比较方式', key: 'compare', onHeaderCell: () => layerHeaderCell(group, 'metric'), render: (_: unknown, r: any) => <Text type="secondary">{getMetricInfo(r.metric).short}</Text> },
                    { title: '平均分', dataIndex: 'mean', key: 'mean', width: 100, onHeaderCell: () => layerHeaderCell(group, 'metric'), render: (v: number) => v != null ? <Text strong>{(v * 100).toFixed(1)}%</Text> : '-' },
                    { title: '最低 / 最高', key: 'range', width: 120, onHeaderCell: () => layerHeaderCell(group, 'metric'), render: (_: unknown, r: any) => r.min != null ? `${(r.min*100).toFixed(1)}% ~ ${(r.max*100).toFixed(1)}%` : '-' },
                    { title: '通过率', dataIndex: 'pass_rate', key: 'pass_rate', width: 100, onHeaderCell: () => layerHeaderCell(group, 'metric'), render: (v: number) => v != null ? <Text style={{ color: passRateColor(v) }}>{(v*100).toFixed(1)}%</Text> : '-' },
                  ]} />
                </Card>
              );
            })}
          </Space>
        </Card>

        <Card title="基本信息">
          <Descriptions column={2} size="small">
            <Descriptions.Item label="评测名称">{evalTask?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="数据集">{evalTask?.dataset?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="场景">{evalTask?.scenario?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="评测 LLM">{evalTask?.llm_config?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="执行时间">{evalTask?.started_at ? new Date(evalTask.started_at).toLocaleString('zh-CN') : '-'}</Descriptions.Item>
            <Descriptions.Item label="耗时">{duration !== null ? `${duration} 秒` : '-'}</Descriptions.Item>
          </Descriptions>
        </Card>
      </div>
    );
  };

  const renderDrawer = () => {
    if (!selectedRow) return null;
    const scores = selectedRow.metric_scores || {};
    const scoreGroups = groupMetricNames(Object.keys(scores));

    return (
      <>
        <div style={{ marginBottom: 16 }}>
          {selectedRow.is_pass === true && <Tag color="success" style={{ fontSize: 14, padding: '4px 12px' }}>✓ 该条数据通过评测</Tag>}
          {selectedRow.is_pass === false && <Tag color="error" style={{ fontSize: 14, padding: '4px 12px' }}>✗ 该条数据未通过评测</Tag>}
          {selectedRow.error && <Tag color="warning" style={{ fontSize: 14, padding: '4px 12px' }}>⚠ 评测异常</Tag>}
        </div>

        <Title level={5}>📊 指标评分详情</Title>
        <Alert message="LLM 类指标由平台原生 Judge 调用当前配置的评测模型打分并返回理由；确定性指标由后端代码直接计算。" type="info" showIcon style={{ marginBottom: 12 }} />

        <Space direction="vertical" style={{ width: '100%' }} size={12}>
          {scoreGroups.map((group) => (
            <Card
              key={group.key}
              size="small"
              title={
                <Space size={6} wrap>
                  <Tag color={group.color}>{group.name}</Tag>
                  <Text type="secondary" style={{ fontSize: 12 }}>（{group.description}）</Text>
                </Space>
              }
            >
              <Table
                rowKey="metric"
                dataSource={group.metrics.map((metric) => {
                  const result = scores[metric];
                  const info = getMetricInfo(metric);
                  const { display, color, explain } = formatScore(result.score, metric);
                  return { metric, info, result, display, color, explain };
                })}
                pagination={false}
                size="small"
                bordered
                columns={[
                  {
                    title: '指标', width: 140, dataIndex: 'metric',
                    onHeaderCell: () => layerHeaderCell(group, 'metric'),
                    render: (_: unknown, r: any) => (
                      <Space size={4} align="start">
                        <div>
                          <Text strong>{r.info.shortName || r.metric}</Text>
                          <div style={{ fontSize: 11, color: '#999', lineHeight: 1.3, marginTop: 2 }}>{r.info.short}</div>
                        </div>
                        {metricHelp(r.metric)}
                      </Space>
                    ),
                  },
                  {
                    title: '得分', width: 90, dataIndex: 'display',
                    onHeaderCell: () => layerHeaderCell(group, 'metric'),
                    render: (_: unknown, r: any) => (
                      <Text style={{ color: r.color, fontWeight: 600, fontSize: 15 }}>{r.display}</Text>
                    ),
                  },
                  {
                    title: '等级', width: 70, dataIndex: 'explain',
                    onHeaderCell: () => layerHeaderCell(group, 'metric'),
                    render: (_: unknown, r: any) => {
                      const level = r.explain.split('（')[0];
                      const tagColor = r.color === '#52c41a' ? 'success' : r.color === '#ff4d4f' ? 'error' : r.color === '#faad14' ? 'warning' : 'processing';
                      return <Tag color={tagColor}>{level}</Tag>;
                    },
                  },
                  {
                    title: '评判标准', dataIndex: 'subjects', ellipsis: false,
                    onHeaderCell: () => layerHeaderCell(group, 'metric'),
                    render: (_: unknown, r: any) => (
                      <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.5 }}>{r.info.subjects || '自定义指标'}</Text>
                    ),
                  },
                  {
                    title: group.key === 'retrieval_unit' ? '计算说明' : 'LLM 评判理由',
                    dataIndex: 'reason',
                    onHeaderCell: () => layerHeaderCell(group, 'metric'),
                    render: (_: unknown, r: any) => (
                      r.result.reason
                        ? <Text style={{ fontSize: 12, lineHeight: 1.5 }}>{r.result.reason}</Text>
                        : <Text type="secondary" style={{ fontSize: 12 }}>未返回理由</Text>
                    ),
                  },
                ]}
              />
            </Card>
          ))}
        </Space>

        <Divider />

        <Title level={5}>📄 原始数据</Title>
        {selectedRow.dataset_row?.data ? (
          <DataFieldsView data={selectedRow.dataset_row.data} />
        ) : <Text type="secondary">无原始数据</Text>}

        {selectedRow.error && (
          <>
            <Divider />
            <Title level={5} style={{ color: '#ff4d4f' }}>⚠ 错误信息</Title>
            <Paragraph type="danger" style={{ whiteSpace: 'pre-wrap' }}>{selectedRow.error}</Paragraph>
          </>
        )}

        <div style={{ marginTop: 16 }}>
          <Text type="secondary">执行耗时: {selectedRow.execution_time_ms ?? '-'}ms</Text>
        </div>
      </>
    );
  };

  return (
    <>
      <Spin spinning={loading}>
        <Space style={{ marginBottom: 16 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/evaluations')}>返回评测列表</Button>
          <Title level={4} style={{ margin: 0 }}>评测报告</Title>
        </Space>

        <Tabs defaultActiveKey="overview" items={[
          { key: 'overview', label: '评测总览', children: renderOverview() },
          {
            key: 'detail', label: '逐条明细',
            children: (
              <div>
                <Space style={{ marginBottom: 16 }}>
                  <Text>状态筛选:</Text>
                  <Select style={{ width: 140 }} allowClear placeholder="全部" value={statusFilter}
                    onChange={(v) => { setStatusFilter(v); setPage(1); }}
                    options={[{ label: '通过', value: 'pass' }, { label: '不通过', value: 'fail' }, { label: '错误', value: 'error' }]}
                  />
                </Space>
                <Table rowKey="id" loading={rowsLoading} columns={detailColumns} dataSource={rows} scroll={{ x: 'max-content' }}
                  pagination={{ current: page, pageSize, total: rowsTotal, showSizeChanger: true, showTotal: (t) => `共 ${t} 条`,
                    onChange: (p, ps) => { setPage(p); setPageSize(ps); },
                  }}
                />
                <Drawer title={`第 ${selectedRow?.row_index ?? '-'} 条评测详情`} open={drawerOpen}
                  onClose={() => { setDrawerOpen(false); setSelectedRow(null); }} width={680}
                >
                  <Spin spinning={detailLoading}>{renderDrawer()}</Spin>
                </Drawer>
              </div>
            ),
          },
        ]} />
      </Spin>
      <MetricHelpDrawer open={!!helpMetric} metricName={helpMetric} onClose={() => setHelpMetric(null)} />
    </>
  );
};

export default ReportDetailPage;

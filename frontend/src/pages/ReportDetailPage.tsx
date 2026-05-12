import React, { useEffect, useState, useCallback } from 'react';
import {
  Tabs, Table, Card, Row, Col, Statistic, Tag, Button, Space, Select,
  Descriptions, Drawer, Typography, Spin, message, Progress, Divider, Tooltip, Alert, Modal, Input,
} from 'antd';
import {
  CheckCircleOutlined, CloseCircleOutlined, ExclamationCircleOutlined,
  ArrowLeftOutlined, QuestionCircleOutlined, InfoCircleOutlined, DatabaseOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate } from 'react-router-dom';
import type { ReportSummary, EvalRowResult, PaginatedResponse, Dataset, DatasetRow, EvalTask, ReportCompareResponse, ReportCompareRowItem } from '../types';
import * as api from '../services/api';
import { MetricHelpDrawer, MetricHelpIcon } from '../components/MetricHelpDrawer';
import { getMetricInfo, groupMetricNames, type MetricLayer } from '../utils/metricLayers';

const { Title, Text, Paragraph } = Typography;

const HelpTitle: React.FC<{ label: string; tip: string }> = ({ label, tip }) => (
  <Space size={4}>
    <span>{label}</span>
    <Tooltip title={tip}>
      <QuestionCircleOutlined style={{ color: '#8c8c8c', fontSize: 12 }} />
    </Tooltip>
  </Space>
);

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
  const [reviewSaving, setReviewSaving] = useState(false);
  const [datasetModalOpen, setDatasetModalOpen] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<Dataset | null>(null);
  const [datasetRows, setDatasetRows] = useState<DatasetRow[]>([]);
  const [datasetRowsTotal, setDatasetRowsTotal] = useState(0);
  const [datasetPage, setDatasetPage] = useState(1);
  const [datasetPageSize, setDatasetPageSize] = useState(10);
  const [datasetPreviewLoading, setDatasetPreviewLoading] = useState(false);
  const [previewRow, setPreviewRow] = useState<DatasetRow | null>(null);
  const [compareTasks, setCompareTasks] = useState<EvalTask[]>([]);
  const [baselineEvalId, setBaselineEvalId] = useState<number | undefined>(undefined);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState<ReportCompareResponse | null>(null);
  const [manualReview, setManualReview] = useState({
    manual_status: undefined as string | undefined,
    manual_score: undefined as number | undefined,
    manual_tags: '',
    manual_note: '',
  });

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

  useEffect(() => {
    const loadCompareTasks = async () => {
      if (!summary?.eval_task) return;
      try {
        const tasks: EvalTask[] = await api.listEvaluations();
        setCompareTasks(
          (Array.isArray(tasks) ? tasks : [])
            .filter((task) =>
              task.id !== evalId &&
              task.status === 'completed' &&
              task.dataset_id === summary.eval_task.dataset_id &&
              task.scenario_id === summary.eval_task.scenario_id
            )
        );
      } catch {
        setCompareTasks([]);
      }
    };
    loadCompareTasks();
  }, [evalId, summary?.eval_task]);

  const loadCompare = async (nextBaselineEvalId: number) => {
    setBaselineEvalId(nextBaselineEvalId);
    setCompareLoading(true);
    try {
      setCompareData(await api.compareReport(evalId, nextBaselineEvalId));
    } catch (error: any) {
      setCompareData(null);
      const detail = error?.response?.data?.detail;
      message.error(detail || '加载基线对比失败');
    } finally {
      setCompareLoading(false);
    }
  };

  const openDetail = async (row: EvalRowResult) => {
    setDrawerOpen(true); setDetailLoading(true);
    try {
      const detail = await api.getReportRowDetail(evalId, row.id);
      setSelectedRow(detail);
      setManualReview({
        manual_status: detail.manual_status || undefined,
        manual_score: detail.manual_score ?? undefined,
        manual_tags: (detail.manual_tags || []).join(', '),
        manual_note: detail.manual_note || '',
      });
    }
    catch {
      setSelectedRow(row);
      setManualReview({
        manual_status: row.manual_status || undefined,
        manual_score: row.manual_score ?? undefined,
        manual_tags: (row.manual_tags || []).join(', '),
        manual_note: row.manual_note || '',
      });
    } finally { setDetailLoading(false); }
  };

  const saveManualReview = async () => {
    if (!selectedRow) return;
    setReviewSaving(true);
    try {
      const payload = {
        manual_status: manualReview.manual_status || null,
        manual_score: manualReview.manual_score ?? null,
        manual_tags: manualReview.manual_tags
          ? manualReview.manual_tags.split(',').map((item) => item.trim()).filter(Boolean)
          : [],
        manual_note: manualReview.manual_note || null,
      };
      const updated = await api.updateReportRowReview(evalId, selectedRow.id, payload);
      setSelectedRow(updated);
      setRows((prev) => prev.map((item) => item.id === updated.id ? updated : item));
      await fetchSummary();
      message.success('人工复核已保存');
    } catch {
      message.error('保存人工复核失败');
    } finally {
      setReviewSaving(false);
    }
  };

  const passRateColor = (r: number) => r >= 0.8 ? '#52c41a' : r >= 0.6 ? '#faad14' : '#ff4d4f';
  const deltaColor = (v?: number | null) => v == null ? '#999' : v > 0 ? '#52c41a' : v < 0 ? '#ff4d4f' : '#666';
  const formatPercent = (v?: number | null) => v == null ? '-' : `${(v * 100).toFixed(1)}%`;
  const formatDelta = (v?: number | null) => {
    if (v == null) return '-';
    const prefix = v > 0 ? '+' : '';
    return `${prefix}${(v * 100).toFixed(1)}%`;
  };
  const metricNames = summary ? Object.keys(summary.metric_summary || {}) : [];
  const evalTask = summary?.eval_task;
  const metricHelp = (metric: string) => (
    <MetricHelpIcon metricName={metric} onClick={() => setHelpMetric(metric)} />
  );
  const datasetUrl = evalTask?.dataset_id ? `/datasets/${evalTask.dataset_id}` : undefined;
  const getDatasetRowUrl = (row?: { dataset_row?: DatasetRow | null; row_index?: number | null } | null) => {
    const datasetId = row?.dataset_row?.dataset_id || evalTask?.dataset_id;
    if (!datasetId) return undefined;
    const params = new URLSearchParams();
    if (row?.dataset_row?.id) params.set('rowId', String(row.dataset_row.id));
    const rowIndex = row?.dataset_row?.row_index ?? row?.row_index;
    if (rowIndex !== undefined && rowIndex !== null) params.set('rowIndex', String(rowIndex));
    const query = params.toString();
    return `/datasets/${datasetId}${query ? `?${query}` : ''}`;
  };
  const loadDatasetPreview = useCallback(async (datasetId: number, nextPage = datasetPage, nextPageSize = datasetPageSize) => {
    setDatasetPreviewLoading(true);
    try {
      const [datasetData, rowsData] = await Promise.all([
        api.getDataset(datasetId),
        api.listDatasetRows(datasetId, nextPage, nextPageSize),
      ]);
      setDatasetPreview(datasetData);
      setDatasetRows(rowsData.items || []);
      setDatasetRowsTotal(rowsData.total || 0);
    } catch {
      message.error('加载数据集预览失败');
    } finally {
      setDatasetPreviewLoading(false);
    }
  }, [datasetPage, datasetPageSize]);

  const openDatasetPreview = async (row?: EvalRowResult | null) => {
    const datasetId = row?.dataset_row?.dataset_id || evalTask?.dataset_id;
    if (!datasetId) return;
    const nextPageSize = 10;
    const rowIndex = row?.dataset_row?.row_index ?? row?.row_index;
    const nextPage = rowIndex !== undefined && rowIndex !== null
      ? Math.floor(rowIndex / nextPageSize) + 1
      : 1;
    setPreviewRow(row?.dataset_row || null);
    setDatasetPage(nextPage);
    setDatasetPageSize(nextPageSize);
    setDatasetModalOpen(true);
    await loadDatasetPreview(datasetId, nextPage, nextPageSize);
  };

  const detailColumns = [
    { title: '#', dataIndex: 'row_index', key: 'row_index', width: 50 },
    {
      title: '状态', key: 'status', width: 70,
      render: (_: unknown, r: EvalRowResult) =>
        r.error ? <Tag color="warning">异常</Tag> :
        r.is_pass === true ? <Tag color="success">通过</Tag> :
        r.is_pass === false ? <Tag color="error">不通过</Tag> : <Tag>-</Tag>,
    },
    {
      title: '人工复核',
      key: 'manual_status',
      width: 110,
      render: (_: unknown, r: EvalRowResult) => {
        const colorMap: Record<string, string> = {
          pass: 'success',
          fail: 'error',
          needs_fix: 'warning',
          needs_review: 'processing',
        };
        const labelMap: Record<string, string> = {
          pass: '人工通过',
          fail: '人工驳回',
          needs_fix: '需修复',
          needs_review: '待复核',
        };
        return r.manual_status ? <Tag color={colorMap[r.manual_status] || 'default'}>{labelMap[r.manual_status] || r.manual_status}</Tag> : <Text type="secondary">未复核</Text>;
      },
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
      title: '操作', key: 'actions', width: 150,
      render: (_: unknown, r: EvalRowResult) => {
        const rowUrl = getDatasetRowUrl(r);
        return (
          <Space size={4}>
            <Button type="link" size="small" onClick={() => openDetail(r)}>详情</Button>
            {rowUrl && (
              <Button type="link" size="small" onClick={() => openDatasetPreview(r)}>
                原始行
              </Button>
            )}
          </Space>
        );
      },
    },
  ];

  const renderOverview = () => {
    if (!summary) return null;
    const metricSummary = summary.metric_summary || {};
    const metricGroups = groupMetricNames(Object.keys(metricSummary));
    const duration = evalTask?.started_at && evalTask?.finished_at
      ? Math.round((new Date(evalTask.finished_at).getTime() - new Date(evalTask.started_at).getTime()) / 1000) : null;
    const evaluatedRows = evalTask?.total_rows ?? summary.total_count;
    const currentDatasetRows = evalTask?.dataset?.row_count;
    const datasetChangedAfterRun =
      typeof currentDatasetRows === 'number' &&
      typeof evaluatedRows === 'number' &&
      currentDatasetRows !== evaluatedRows;

    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title={<HelpTitle label="样本通过率" tip="按样本维度统计：一条样本需要满足本次场景下所有参与准入判断的指标阈值，才会计为样本通过。" />}
                value={(summary.pass_rate * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: passRateColor(summary.pass_rate) }}
              />
            </Card>
          </Col>
          <Col span={6}><Card><Statistic title="评测样本数" value={summary.total_count} /></Card></Col>
          <Col span={6}><Card><Statistic title="样本通过" value={summary.pass_count} valueStyle={{ color: '#52c41a' }} /></Card></Col>
          <Col span={6}><Card><Statistic title="样本不通过" value={summary.fail_count} valueStyle={{ color: '#ff4d4f' }} /></Card></Col>
        </Row>
        {summary.manual_review_summary && (
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title={<HelpTitle label="已人工复核" tip="人工已给出复核结论的样本数，不参与自动样本通过率计算。" />}
                  value={summary.manual_review_summary.reviewed_count || 0}
                />
              </Card>
            </Col>
            <Col span={6}><Card><Statistic title="人工判定通过" value={summary.manual_review_summary.manual_pass_count || 0} valueStyle={{ color: '#52c41a' }} /></Card></Col>
            <Col span={6}><Card><Statistic title="人工判定驳回" value={summary.manual_review_summary.manual_fail_count || 0} valueStyle={{ color: '#ff4d4f' }} /></Card></Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title={<HelpTitle label="待修复/待复核" tip="人工标记为需要修复或仍需继续复核的样本数，用于后续坏例沉淀和跟进。" />}
                  value={(summary.manual_review_summary.manual_needs_fix_count || 0) + (summary.manual_review_summary.manual_needs_review_count || 0)}
                  valueStyle={{ color: '#faad14' }}
                />
              </Card>
            </Col>
          </Row>
        )}

        <Card title="各指标得分概览" extra={<Text type="secondary">指标维度统计，分数范围 0~1；通过率按单个指标阈值单独计算</Text>} style={{ marginBottom: 24 }}>
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
                    {
                      title: <HelpTitle label="指标通过率" tip="按单个指标统计：该指标得分达到阈值的样本数 / 该指标成功出分的样本数。它和顶部样本通过率不是同一个统计口径。" />,
                      dataIndex: 'pass_rate',
                      key: 'pass_rate',
                      width: 120,
                      onHeaderCell: () => layerHeaderCell(group, 'metric'),
                      render: (v: number) => v != null ? <Text style={{ color: passRateColor(v) }}>{(v*100).toFixed(1)}%</Text> : '-',
                    },
                  ]} />
                </Card>
              );
            })}
          </Space>
        </Card>

        <Card title="基本信息">
          {datasetChangedAfterRun && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message="当前数据集行数与本次评测行数不一致"
              description={`这份报告是历史执行结果：本次评测实际执行 ${evaluatedRows} 条；当前数据集已有 ${currentDatasetRows} 条。新增或删除的数据不会自动进入旧报告，请在评测执行页克隆该任务并重新执行。`}
            />
          )}
          <Descriptions column={2} size="small">
            <Descriptions.Item label="评测名称">{evalTask?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="数据集">
              {datasetUrl ? (
                <Button
                  type="link"
                  size="small"
                  icon={<DatabaseOutlined />}
                  style={{ padding: 0, height: 'auto' }}
                  onClick={() => openDatasetPreview()}
                >
                  {evalTask?.dataset?.name || `数据集 #${evalTask?.dataset_id}`}
                </Button>
              ) : evalTask?.dataset?.name || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="本次评测行数">{evaluatedRows ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="当前数据集行数">{currentDatasetRows ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="场景">{evalTask?.scenario_snapshot?.name || evalTask?.scenario?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="评测 LLM">{evalTask?.llm_config?.name || '-'}</Descriptions.Item>
            <Descriptions.Item label="执行时间">{evalTask?.started_at ? new Date(evalTask.started_at).toLocaleString('zh-CN') : '-'}</Descriptions.Item>
            <Descriptions.Item label="耗时">{duration !== null ? `${duration} 秒` : '-'}</Descriptions.Item>
          </Descriptions>
        </Card>
      </div>
    );
  };

  const statusLabel = (status?: string | null) => {
    const labelMap: Record<string, string> = { pass: '通过', fail: '不通过', error: '异常', unknown: '未知' };
    const colorMap: Record<string, string> = { pass: 'success', fail: 'error', error: 'warning', unknown: 'default' };
    return status ? <Tag color={colorMap[status] || 'default'}>{labelMap[status] || status}</Tag> : <Text type="secondary">-</Text>;
  };

  const renderCompareRows = (title: string, rows: ReportCompareRowItem[], color: string) => (
    <Card
      size="small"
      title={<Space><Tag color={color}>{title}</Tag><Text type="secondary">{rows.length} 条</Text></Space>}
      style={{ marginBottom: 16 }}
    >
      <Table
        rowKey={(record) => `${title}-${record.dataset_row_id}`}
        size="small"
        dataSource={rows}
        pagination={{ pageSize: 8 }}
        columns={[
          { title: '#', dataIndex: 'row_index', key: 'row_index', width: 70 },
          { title: '基线状态', dataIndex: 'baseline_status', key: 'baseline_status', width: 100, render: statusLabel },
          { title: '当前状态', dataIndex: 'current_status', key: 'current_status', width: 100, render: statusLabel },
          {
            title: '主要指标变化',
            key: 'metric_delta',
            render: (_: unknown, record: ReportCompareRowItem) => {
              const entries = Object.entries(record.metric_deltas || {}).slice(0, 4);
              if (entries.length === 0) return <Text type="secondary">-</Text>;
              return (
                <Space size={[4, 4]} wrap>
                  {entries.map(([metric, item]) => (
                    <Tag key={metric} color={item.delta == null ? 'default' : item.delta >= 0 ? 'green' : 'red'}>
                      {getMetricInfo(metric).shortName || metric}: {formatDelta(item.delta)}
                    </Tag>
                  ))}
                </Space>
              );
            },
          },
          {
            title: '操作',
            key: 'actions',
            width: 90,
            render: (_: unknown, record: ReportCompareRowItem) => (
              record.current_result_id ? (
                <Button
                  type="link"
                  size="small"
                  onClick={async () => {
                    const row = rows.find((item) => item.current_result_id === record.current_result_id);
                    if (row?.current_result_id) {
                      await openDetail({
                        id: row.current_result_id,
                        eval_task_id: evalId,
                        row_index: row.row_index,
                        metric_scores: {},
                        is_pass: null,
                        execution_time_ms: null,
                        error: null,
                        dataset_row: row.dataset_row || undefined,
                        created_at: new Date().toISOString(),
                      } as EvalRowResult);
                    }
                  }}
                >
                  详情
                </Button>
              ) : '-'
            ),
          },
        ]}
      />
    </Card>
  );

  const renderCompare = () => (
    <div>
      <Card style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: '100%' }} size={12}>
          <Space wrap>
            <Text strong>选择基线任务</Text>
            <Select
              style={{ width: 360 }}
              placeholder="选择同数据集、同场景的历史评测任务"
              value={baselineEvalId}
              loading={compareLoading}
              options={compareTasks.map((task) => ({
                label: `${task.name} · ${new Date(task.created_at).toLocaleString('zh-CN')}`,
                value: task.id,
              }))}
              onChange={loadCompare}
            />
          </Space>
          <Text type="secondary">
            本期基线对比仅支持同一数据集、同一场景、同一指标集，样本按数据行 ID 精确匹配。
          </Text>
          {compareTasks.length === 0 && (
            <Alert type="info" showIcon message="暂无可对比的历史评测任务" />
          )}
        </Space>
      </Card>

      <Spin spinning={compareLoading}>
        {compareData && (
          <>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title={<HelpTitle label="样本通过率变化" tip="对比两次评测的整体样本通过率，按样本通过/不通过结果计算。" />}
                    value={formatDelta(compareData.summary_delta.pass_rate_delta)}
                    valueStyle={{ color: deltaColor(compareData.summary_delta.pass_rate_delta) }}
                  />
                  <Text type="secondary">
                    基线样本 {formatPercent(compareData.summary_delta.baseline_pass_rate)} / 当前样本 {formatPercent(compareData.summary_delta.current_pass_rate)}
                  </Text>
                </Card>
              </Col>
              <Col span={6}><Card><Statistic title="新增失败" value={compareData.row_changes.new_failures?.length || 0} valueStyle={{ color: '#ff4d4f' }} /></Card></Col>
              <Col span={6}><Card><Statistic title="已修复" value={compareData.row_changes.fixed?.length || 0} valueStyle={{ color: '#52c41a' }} /></Card></Col>
              <Col span={6}><Card><Statistic title="持续失败" value={compareData.row_changes.still_failing?.length || 0} valueStyle={{ color: '#faad14' }} /></Card></Col>
            </Row>

            <Card title="指标变化" style={{ marginBottom: 16 }}>
              <Table
                rowKey="metric"
                size="small"
                dataSource={compareData.metric_deltas}
                pagination={false}
                columns={[
                  { title: '指标', dataIndex: 'metric', key: 'metric', render: (metric: string) => getMetricInfo(metric).shortName || metric },
                  { title: '基线均分', dataIndex: 'baseline_mean', key: 'baseline_mean', render: formatPercent },
                  { title: '当前均分', dataIndex: 'current_mean', key: 'current_mean', render: formatPercent },
                  { title: '均分变化', dataIndex: 'mean_delta', key: 'mean_delta', render: (v: number | null) => <Text style={{ color: deltaColor(v) }}>{formatDelta(v)}</Text> },
                  { title: <HelpTitle label="基线指标通过率" tip="基线评测中，该单个指标达到阈值的样本占比。" />, dataIndex: 'baseline_pass_rate', key: 'baseline_pass_rate', render: formatPercent },
                  { title: <HelpTitle label="当前指标通过率" tip="当前评测中，该单个指标达到阈值的样本占比。" />, dataIndex: 'current_pass_rate', key: 'current_pass_rate', render: formatPercent },
                  { title: <HelpTitle label="指标通过率变化" tip="当前指标通过率减去基线指标通过率。" />, dataIndex: 'pass_rate_delta', key: 'pass_rate_delta', render: (v: number | null) => <Text style={{ color: deltaColor(v) }}>{formatDelta(v)}</Text> },
                ]}
              />
            </Card>

            {renderCompareRows('新增失败', compareData.row_changes.new_failures || [], 'red')}
            {renderCompareRows('已修复', compareData.row_changes.fixed || [], 'green')}
            {renderCompareRows('持续失败', compareData.row_changes.still_failing || [], 'orange')}
            {renderCompareRows('新增异常', compareData.row_changes.new_errors || [], 'warning')}
            {renderCompareRows('持续通过', compareData.row_changes.still_passing || [], 'blue')}
          </>
        )}
      </Spin>
    </div>
  );

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

        <Card size="small" title="🧑 人工复核" style={{ marginBottom: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }} size={12}>
            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>人工结论</Text>
              <Select
                style={{ width: 220 }}
                allowClear
                placeholder="请选择人工结论"
                value={manualReview.manual_status}
                onChange={(value) => setManualReview((prev) => ({ ...prev, manual_status: value }))}
                options={[
                  { label: '人工通过', value: 'pass' },
                  { label: '人工驳回', value: 'fail' },
                  { label: '需要修复', value: 'needs_fix' },
                  { label: '待复核', value: 'needs_review' },
                ]}
              />
            </div>
            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>人工分数（可选）</Text>
              <Input
                style={{ width: 220 }}
                placeholder="0 ~ 1，例如 0.8"
                value={manualReview.manual_score as any}
                onChange={(e) => setManualReview((prev) => ({
                  ...prev,
                  manual_score: e.target.value === '' ? undefined : Number(e.target.value),
                }))}
              />
            </div>
            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>问题标签</Text>
              <Input
                placeholder="用逗号分隔，例如：检索漏召回, 回答不完整"
                value={manualReview.manual_tags}
                onChange={(e) => setManualReview((prev) => ({ ...prev, manual_tags: e.target.value }))}
              />
            </div>
            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>复核备注</Text>
              <Input.TextArea
                autoSize={{ minRows: 3, maxRows: 6 }}
                placeholder="写下为什么自动结论不可信，或这条样本后续要怎么处理"
                value={manualReview.manual_note}
                onChange={(e) => setManualReview((prev) => ({ ...prev, manual_note: e.target.value }))}
              />
            </div>
            <Space align="center">
              <Button type="primary" loading={reviewSaving} onClick={saveManualReview}>保存人工复核</Button>
              {selectedRow.reviewed_at && <Text type="secondary">上次保存：{new Date(selectedRow.reviewed_at).toLocaleString('zh-CN')}</Text>}
            </Space>
          </Space>
        </Card>

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

        <Space align="center" style={{ width: '100%', justifyContent: 'space-between', marginBottom: 8 }}>
          <Title level={5} style={{ margin: 0 }}>📄 原始数据</Title>
          {getDatasetRowUrl(selectedRow) && (
            <Button
              size="small"
              icon={<DatabaseOutlined />}
              onClick={() => openDatasetPreview(selectedRow)}
            >
              在数据集中定位
            </Button>
          )}
        </Space>
        {selectedRow.dataset_row?.data ? (
          <DataFieldsView data={selectedRow.dataset_row.data} />
        ) : <Text type="secondary">无原始数据</Text>}

        {selectedRow.endpoint_trace && (
          <>
            <Divider />
            <Title level={5}>接口调用追踪</Title>
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <Space wrap>
                <Tag color={selectedRow.endpoint_trace.status === 'success' ? 'green' : 'red'}>
                  {selectedRow.endpoint_trace.status === 'success' ? '调用成功' : '调用失败'}
                </Tag>
                {selectedRow.endpoint_trace.status_code && <Text type="secondary">HTTP {selectedRow.endpoint_trace.status_code}</Text>}
                {selectedRow.endpoint_trace.latency_ms && <Text type="secondary">{selectedRow.endpoint_trace.latency_ms}ms</Text>}
              </Space>
              {selectedRow.endpoint_trace.extracted_fields && (
                <Card size="small" title="解析后的评测字段">
                  <DataFieldsView data={selectedRow.endpoint_trace.extracted_fields} />
                </Card>
              )}
              {selectedRow.endpoint_trace.mapping_errors && Object.keys(selectedRow.endpoint_trace.mapping_errors).length > 0 && (
                <Alert
                  type="warning"
                  showIcon
                  message="字段映射提示"
                  description={JSON.stringify(selectedRow.endpoint_trace.mapping_errors)}
                />
              )}
              {selectedRow.endpoint_trace.request_body && (
                <Card size="small" title="请求体">
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', maxHeight: 220, overflow: 'auto' }}>
                    {JSON.stringify(selectedRow.endpoint_trace.request_body, null, 2)}
                  </pre>
                </Card>
              )}
              {selectedRow.endpoint_trace.raw_response && (
                <Card size="small" title="原始响应">
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', maxHeight: 260, overflow: 'auto' }}>
                    {selectedRow.endpoint_trace.raw_response}
                  </pre>
                </Card>
              )}
            </Space>
          </>
        )}

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

  const renderDatasetPreviewValue = (value: any) => {
    if (value === null || value === undefined) return <Text type="secondary">-</Text>;
    const text = typeof value === 'string' ? value : JSON.stringify(value);
    return (
      <Text style={{ fontSize: 12 }}>
        {text.length > 80 ? `${text.slice(0, 80)}...` : text}
      </Text>
    );
  };

  const datasetFieldNames = datasetPreview?.field_schema?.length
    ? datasetPreview.field_schema.map((field) => field.name)
    : Array.from(new Set(datasetRows.flatMap((row) => Object.keys(row.data || {}))));

  const datasetPreviewColumns = [
    { title: '#', dataIndex: 'row_index', key: 'row_index', width: 60 },
    ...datasetFieldNames.map((fieldName) => ({
      title: FIELD_META[fieldName]?.label || fieldName,
      key: fieldName,
      width: 180,
      ellipsis: true,
      render: (_: unknown, record: DatasetRow) => renderDatasetPreviewValue(record.data?.[fieldName]),
    })),
  ];

  return (
    <>
      <style>
        {`
          .dataset-row-highlight > td {
            background: #fff7e6 !important;
            box-shadow: inset 3px 0 0 #faad14;
          }
        `}
      </style>
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
          { key: 'compare', label: '基线对比', children: renderCompare() },
        ]} />
        <Modal
          title={
            <Space>
              <DatabaseOutlined />
              <span>{previewRow ? `原始数据行 #${previewRow.row_index}` : '数据集预览'}</span>
            </Space>
          }
          open={datasetModalOpen}
          onCancel={() => {
            setDatasetModalOpen(false);
            setPreviewRow(null);
          }}
          width="88vw"
          style={{ top: 32 }}
          footer={
            <Space>
              {datasetUrl && (
                <Button onClick={() => navigate(previewRow ? getDatasetRowUrl({ dataset_row: previewRow }) || datasetUrl : datasetUrl)}>
                  打开数据集页面
                </Button>
              )}
              <Button type="primary" onClick={() => setDatasetModalOpen(false)}>关闭</Button>
            </Space>
          }
        >
          <Spin spinning={datasetPreviewLoading}>
            {datasetPreview && (
              <Descriptions size="small" column={3} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="数据集">{datasetPreview.name}</Descriptions.Item>
                <Descriptions.Item label="样本类型">{datasetPreview.sample_type}</Descriptions.Item>
                <Descriptions.Item label="数据行数">{datasetPreview.row_count}</Descriptions.Item>
              </Descriptions>
            )}
            {previewRow?.data && (
              <Card size="small" title="当前原始行内容" style={{ marginBottom: 16 }}>
                <DataFieldsView data={previewRow.data} />
              </Card>
            )}
            <Table
              rowKey="id"
              size="small"
              columns={datasetPreviewColumns}
              dataSource={datasetRows}
              scroll={{ x: 'max-content', y: 360 }}
              rowClassName={(record) => previewRow?.id === record.id ? 'dataset-row-highlight' : ''}
              onRow={(record) => ({ onClick: () => setPreviewRow(record) })}
              pagination={{
                current: datasetPage,
                pageSize: datasetPageSize,
                total: datasetRowsTotal,
                showSizeChanger: true,
                showTotal: (t) => `共 ${t} 条`,
                onChange: async (nextPage, nextPageSize) => {
                  setDatasetPage(nextPage);
                  setDatasetPageSize(nextPageSize);
                  if (evalTask?.dataset_id) {
                    await loadDatasetPreview(evalTask.dataset_id, nextPage, nextPageSize);
                  }
                },
              }}
            />
          </Spin>
        </Modal>
      </Spin>
      <MetricHelpDrawer open={!!helpMetric} metricName={helpMetric} onClose={() => setHelpMetric(null)} />
    </>
  );
};

export default ReportDetailPage;

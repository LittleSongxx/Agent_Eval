import React from 'react';
import { Checkbox, InputNumber, Tag, Space, Typography } from 'antd';
import type { MetricDefinition } from '../types';

const { Text } = Typography;

interface SelectedMetric {
  metric_definition_id: number;
  weight: number;
  pass_threshold: number | null;
}

interface MetricConfigPanelProps {
  availableMetrics: MetricDefinition[];
  selectedMetrics: SelectedMetric[];
  onChange: (metrics: SelectedMetric[]) => void;
}

const MetricConfigPanel: React.FC<MetricConfigPanelProps> = ({
  availableMetrics,
  selectedMetrics,
  onChange,
}) => {
  const isSelected = (metricId: number) =>
    selectedMetrics.some((m) => m.metric_definition_id === metricId);

  const handleToggle = (metricId: number, checked: boolean) => {
    if (checked) {
      onChange([
        ...selectedMetrics,
        { metric_definition_id: metricId, weight: 1.0, pass_threshold: null },
      ]);
    } else {
      onChange(
        selectedMetrics.filter((m) => m.metric_definition_id !== metricId)
      );
    }
  };

  const handleWeightChange = (metricId: number, weight: number) => {
    onChange(
      selectedMetrics.map((m) =>
        m.metric_definition_id === metricId ? { ...m, weight } : m
      )
    );
  };

  const handleThresholdChange = (
    metricId: number,
    pass_threshold: number | null
  ) => {
    onChange(
      selectedMetrics.map((m) =>
        m.metric_definition_id === metricId ? { ...m, pass_threshold } : m
      )
    );
  };

  if (availableMetrics.length === 0) {
    return (
      <div style={{ padding: '16px 0' }}>
        <Text type="secondary">当前场景类型没有可用的评测指标</Text>
      </div>
    );
  }

  return (
    <div style={{ marginTop: 12 }}>
      {availableMetrics.map((metric) => {
        const checked = isSelected(metric.id);
        const selectedItem = selectedMetrics.find(
          (m) => m.metric_definition_id === metric.id
        );

        return (
          <div
            key={metric.id}
            style={{
              padding: '8px 0',
              borderBottom: '1px solid #f0f0f0',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 8,
              }}
            >
              <Checkbox
                checked={checked}
                onChange={(e) => handleToggle(metric.id, e.target.checked)}
              >
                <Space>
                  <Text strong>{metric.display_name}</Text>
                  <Tag color="blue">{metric.category}</Tag>
                </Space>
              </Checkbox>

              {checked && selectedItem && (
                <Space style={{ marginLeft: 'auto' }}>
                  <Text type="secondary">权重:</Text>
                  <InputNumber
                    min={0}
                    max={10}
                    step={0.1}
                    value={selectedItem.weight}
                    onChange={(v) => handleWeightChange(metric.id, v ?? 1)}
                    size="small"
                    style={{ width: 72 }}
                  />
                  <Text type="secondary">通过阈值:</Text>
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.05}
                    value={selectedItem.pass_threshold ?? undefined}
                    onChange={(v) => handleThresholdChange(metric.id, v)}
                    placeholder="可选"
                    size="small"
                    style={{ width: 72 }}
                  />
                </Space>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MetricConfigPanel;

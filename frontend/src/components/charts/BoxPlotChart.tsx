import React from 'react';
import { ResponsiveContainer, ComposedChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Label } from 'recharts';

interface BoxPlotData {
  metric: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers?: number[];
}

interface BoxPlotChartProps {
  data: BoxPlotData[];
  title?: string;
}

export const BoxPlotChart: React.FC<BoxPlotChartProps> = ({ data, title }) => {
  // 转换数据格式为 Recharts 可识别的格式
  const chartData = data.map(item => ({
    name: item.metric,
    // 使用 stacked bar 来模拟箱线图
    lower: item.min,
    q1Range: item.q1 - item.min,
    iqr: item.q3 - item.q1,
    q3Range: item.max - item.q3,
    median: item.median,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const original = data.name;
      const item = data;

      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold mb-2">{original}</p>
          <p className="text-sm">最大值: {(item.lower + item.q1Range + item.iqr + item.q3Range).toFixed(3)}</p>
          <p className="text-sm">Q3 (75%): {(item.lower + item.q1Range + item.iqr).toFixed(3)}</p>
          <p className="text-sm">中位数: {item.median.toFixed(3)}</p>
          <p className="text-sm">Q1 (25%): {(item.lower + item.q1Range).toFixed(3)}</p>
          <p className="text-sm">最小值: {item.lower.toFixed(3)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-96 p-4 bg-white rounded-lg shadow">
      {title && <h3 className="text-lg font-semibold mb-4 text-center">{title}</h3>}
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name">
            <Label value="指标" offset={-40} position="insideBottom" />
          </XAxis>
          <YAxis domain={[0, 1]}>
            <Label value="分数分布" angle={-90} position="insideLeft" />
          </YAxis>
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* 下须 */}
          <Bar dataKey="lower" stackId="box" fill="transparent" />
          {/* Q1 区域 */}
          <Bar dataKey="q1Range" stackId="box" fill="#d0d0d0" name="Q1-Min" />
          {/* IQR (箱体) */}
          <Bar dataKey="iqr" stackId="box" fill="#8884d8" name="IQR (Q1-Q3)" />
          {/* Q3-Max 区域 */}
          <Bar dataKey="q3Range" stackId="box" fill="#d0d0d0" name="Max-Q3" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-4 text-sm text-gray-600 text-center">
        <p>箱体 (蓝色): IQR (Q1-Q3)，包含中间 50% 的数据</p>
        <p>须 (灰色): 最小值到 Q1，Q3 到最大值</p>
      </div>
    </div>
  );
};

export default BoxPlotChart;

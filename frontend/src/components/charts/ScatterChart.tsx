import React from 'react';
import { Scatter, ScatterChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Label } from 'recharts';

interface ScatterPoint {
  x: number;
  y: number;
  name?: string;
}

interface MetricScatterChartProps {
  data: ScatterPoint[];
  xLabel: string;
  yLabel: string;
  title?: string;
  correlationCoefficient?: number;
}

export const MetricScatterChart: React.FC<MetricScatterChartProps> = ({
  data,
  xLabel,
  yLabel,
  title,
  correlationCoefficient
}) => {
  return (
    <div className="w-full h-96 p-4 bg-white rounded-lg shadow">
      {title && (
        <div className="text-center mb-4">
          <h3 className="text-lg font-semibold">{title}</h3>
          {correlationCoefficient !== undefined && (
            <p className="text-sm text-gray-600">
              相关系数: {correlationCoefficient.toFixed(3)}
            </p>
          )}
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name={xLabel}
            domain={[0, 1]}
          >
            <Label value={xLabel} offset={-40} position="insideBottom" />
          </XAxis>
          <YAxis
            type="number"
            dataKey="y"
            name={yLabel}
            domain={[0, 1]}
          >
            <Label value={yLabel} angle={-90} position="insideLeft" />
          </YAxis>
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Legend />
          <Scatter
            name={`${xLabel} vs ${yLabel}`}
            data={data}
            fill="#8884d8"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MetricScatterChart;

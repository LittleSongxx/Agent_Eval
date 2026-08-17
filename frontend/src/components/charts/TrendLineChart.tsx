import React from 'react';
import { Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Label } from 'recharts';

interface TrendPoint {
  iteration: number;
  value: number;
  label?: string;
}

interface TrendLineChartProps {
  data: TrendPoint[];
  title?: string;
  yLabel?: string;
  showImprovement?: boolean;
}

export const TrendLineChart: React.FC<TrendLineChartProps> = ({
  data,
  title,
  yLabel = '分数',
  showImprovement = true
}) => {
  const improvement = data.length > 1
    ? ((data[data.length - 1].value - data[0].value) / data[0].value * 100).toFixed(1)
    : '0';

  return (
    <div className="w-full h-96 p-4 bg-white rounded-lg shadow">
      {title && (
        <div className="text-center mb-4">
          <h3 className="text-lg font-semibold">{title}</h3>
          {showImprovement && data.length > 1 && (
            <p className="text-sm text-gray-600">
              总体提升: {improvement}%
              <span className="ml-2 text-xs">
                ({data[0].value.toFixed(3)} → {data[data.length - 1].value.toFixed(3)})
              </span>
            </p>
          )}
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="iteration"
            type="number"
            domain={['dataMin', 'dataMax']}
          >
            <Label value="迭代次数" offset={-40} position="insideBottom" />
          </XAxis>
          <YAxis
            domain={[0, 1]}
          >
            <Label value={yLabel} angle={-90} position="insideLeft" />
          </YAxis>
          <Tooltip
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={(label: number) => `迭代 ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#8884d8"
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
            name={yLabel}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TrendLineChart;

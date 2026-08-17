import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface RadarChartData {
  metric: string;
  baseline: number;
  optimized: number;
  fullMark: number;
}

interface MetricRadarChartProps {
  data: RadarChartData[];
  title?: string;
}

export const MetricRadarChart: React.FC<MetricRadarChartProps> = ({ data, title }) => {
  return (
    <div className="w-full h-96 p-4 bg-white rounded-lg shadow">
      {title && <h3 className="text-lg font-semibold mb-4 text-center">{title}</h3>}
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" />
          <PolarRadiusAxis angle={90} domain={[0, 1]} />
          <Radar
            name="基线版本"
            dataKey="baseline"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.3}
          />
          <Radar
            name="优化版本"
            dataKey="optimized"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.3}
          />
          <Legend />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MetricRadarChart;

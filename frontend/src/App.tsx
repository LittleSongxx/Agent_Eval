import React, { useState } from 'react';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import {
  SettingOutlined,
  DatabaseOutlined,
  AppstoreOutlined,
  PlayCircleOutlined,
  BarChartOutlined,
  SwapOutlined,
  SlidersOutlined,
  ApiOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';

import ExperimentWorkbenchPage from './pages/ExperimentWorkbenchPage';
import LLMConfigPage from './pages/LLMConfigPage';
import DatasetListPage from './pages/DatasetListPage';
import DatasetDetailPage from './pages/DatasetDetailPage';
import RagDatasetBuilderPage from './pages/RagDatasetBuilderPage';
import ScenarioListPage from './pages/ScenarioListPage';
import MetricListPage from './pages/MetricListPage';
import EndpointTargetPage from './pages/EndpointTargetPage';
import EvaluationPage from './pages/EvaluationPage';
import ReportListPage from './pages/ReportListPage';
import ReportDetailPage from './pages/ReportDetailPage';
import BlindTestPage from './pages/BlindTestPage';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

const menuItems = [
  {
    key: 'workflow',
    type: 'group' as const,
    label: '实验流程',
    children: [
      { key: '/workbench', icon: <ExperimentOutlined />, label: '评测实验工作台' },
    ],
  },
  {
    key: 'resources',
    type: 'group' as const,
    label: '资源管理',
    children: [
      { key: '/datasets', icon: <DatabaseOutlined />, label: '数据管理' },
      { key: '/metrics', icon: <SlidersOutlined />, label: '指标管理' },
      { key: '/endpoint-targets', icon: <ApiOutlined />, label: '被测接口' },
      { key: '/scenarios', icon: <AppstoreOutlined />, label: '场景管理' },
      { key: '/llm-configs', icon: <SettingOutlined />, label: 'LLM配置' },
    ],
  },
  {
    key: 'results',
    type: 'group' as const,
    label: '结果分析',
    children: [
      { key: '/evaluations', icon: <PlayCircleOutlined />, label: '评测执行' },
      { key: '/reports', icon: <BarChartOutlined />, label: '评测报告' },
      { key: '/blind-tests', icon: <SwapOutlined />, label: '人工盲测' },
    ],
  },
];

const App: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  const selectedKey = '/' + location.pathname.split('/')[1];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
          background: '#001529',
        }}
      >
        <Title level={4} style={{ color: '#fff', margin: 0, whiteSpace: 'nowrap' }}>
          AI 评测平台
        </Title>
      </Header>
      <Layout>
        <Sider
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          theme="dark"
          width={200}
          collapsedWidth={80}
        >
          <Menu
            mode="inline"
            theme="dark"
            selectedKeys={[selectedKey]}
            style={{ height: '100%', borderRight: 0, paddingTop: 16 }}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
          />
        </Sider>
        <Layout style={{ padding: 24 }}>
          <Content
            style={{
              background: '#fff',
              padding: 24,
              margin: 0,
              borderRadius: 8,
              minHeight: 280,
            }}
          >
            <Routes>
              <Route path="/workbench" element={<ExperimentWorkbenchPage />} />
              <Route path="/llm-configs" element={<LLMConfigPage />} />
              <Route path="/datasets" element={<DatasetListPage />} />
              <Route path="/datasets/:id" element={<DatasetDetailPage />} />
              <Route path="/datasets/rag-builder" element={<RagDatasetBuilderPage />} />
              <Route path="/metrics" element={<MetricListPage />} />
              <Route path="/endpoint-targets" element={<EndpointTargetPage />} />
              <Route path="/scenarios" element={<ScenarioListPage />} />
              <Route path="/evaluations" element={<EvaluationPage />} />
              <Route path="/blind-tests" element={<BlindTestPage />} />
              <Route path="/reports" element={<ReportListPage />} />
              <Route path="/reports/:id" element={<ReportDetailPage />} />
              <Route path="*" element={<Navigate to="/workbench" replace />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default App;

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { App as AntdApp, ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ConfigProvider
        locale={zhCN}
        theme={{
          algorithm: theme.defaultAlgorithm,
          token: {
            colorPrimary: '#2563eb',
            colorInfo: '#0f766e',
            colorSuccess: '#16a34a',
            colorWarning: '#d97706',
            colorError: '#dc2626',
            colorText: '#172033',
            colorTextSecondary: '#5f6b7a',
            colorBgBase: '#f5f7fb',
            colorBgLayout: '#f3f6fb',
            colorBorder: '#d9e1ec',
            borderRadius: 8,
            boxShadowSecondary: '0 14px 38px rgba(23, 32, 51, 0.08)',
            fontFamily:
              'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          },
          components: {
            Button: {
              controlHeight: 36,
              borderRadius: 7,
              primaryShadow: '0 8px 18px rgba(37, 99, 235, 0.18)',
            },
            Card: {
              borderRadiusLG: 8,
              headerBg: '#ffffff',
              paddingLG: 20,
            },
            Layout: {
              bodyBg: '#f3f6fb',
              headerBg: '#ffffff',
              siderBg: '#0f172a',
            },
            Menu: {
              darkItemBg: '#0f172a',
              darkSubMenuItemBg: '#0f172a',
              darkItemSelectedBg: '#2563eb',
              darkItemHoverBg: 'rgba(255, 255, 255, 0.08)',
              itemBorderRadius: 7,
            },
            Modal: {
              borderRadiusLG: 8,
            },
            Table: {
              headerBg: '#f8fafc',
              headerColor: '#39465a',
              rowHoverBg: '#f5f8ff',
              borderColor: '#e3e9f2',
            },
            Tag: {
              borderRadiusSM: 6,
            },
          },
        }}
      >
        <AntdApp>
          <App />
        </AntdApp>
      </ConfigProvider>
    </BrowserRouter>
  </React.StrictMode>,
);

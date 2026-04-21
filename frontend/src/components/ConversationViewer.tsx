import React from 'react';
import { Tag, Typography } from 'antd';

const { Text } = Typography;

interface ConversationMessage {
  type: string;
  content: string;
  tool_calls?: Array<{ name: string; arguments: any }>;
}

interface ConversationViewerProps {
  messages: ConversationMessage[];
}

const ConversationViewer: React.FC<ConversationViewerProps> = ({ messages }) => {
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return <Text type="secondary">无对话数据</Text>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {messages.map((msg, idx) => {
        const isHuman =
          msg.type === 'human' ||
          msg.type === 'user' ||
          msg.type === 'HumanMessage';
        const isAI =
          msg.type === 'ai' ||
          msg.type === 'assistant' ||
          msg.type === 'AIMessage';
        const isTool =
          msg.type === 'tool' || msg.type === 'ToolMessage';

        let bgColor = '#f5f5f5';
        let label = msg.type;

        if (isHuman) {
          bgColor = '#e6f4ff';
          label = '用户';
        } else if (isAI) {
          bgColor = '#f5f5f5';
          label = 'AI';
        } else if (isTool) {
          bgColor = '#f6ffed';
          label = '工具';
        }

        return (
          <div key={idx}>
            <div
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                backgroundColor: bgColor,
                border: isTool ? '1px solid #b7eb8f' : '1px solid #f0f0f0',
              }}
            >
              <div style={{ marginBottom: 4 }}>
                <Tag
                  color={isHuman ? 'blue' : isAI ? 'default' : 'green'}
                  style={{ marginRight: 4 }}
                >
                  {label}
                </Tag>
              </div>
              <Text
                style={{
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  fontSize: 13,
                }}
              >
                {msg.content}
              </Text>

              {msg.tool_calls && msg.tool_calls.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {msg.tool_calls.map((tc, tcIdx) => (
                    <div key={tcIdx} style={{ marginBottom: 4 }}>
                      <Tag color="purple">{tc.name}</Tag>
                      <pre
                        style={{
                          background: '#fafafa',
                          padding: 8,
                          borderRadius: 4,
                          fontSize: 12,
                          margin: '4px 0 0 0',
                          maxHeight: 120,
                          overflow: 'auto',
                        }}
                      >
                        {typeof tc.arguments === 'string'
                          ? tc.arguments
                          : JSON.stringify(tc.arguments, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ConversationViewer;

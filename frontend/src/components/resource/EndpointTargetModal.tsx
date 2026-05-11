import React, { useEffect, useState } from 'react';
import { Alert, Button, Col, Descriptions, Divider, Form, Input, Modal, Row, Select, Space, Typography, message } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';
import type { EndpointTarget } from '../../types';
import * as api from '../../services/api';

const { TextArea } = Input;
const { Text } = Typography;

const DEFAULT_TEST_INPUT = '请用三句话介绍一下 DeepSeek，并说明它适合做哪些 AI 应用测试。';
const DEFAULT_BODY = JSON.stringify(
  {
    model: 'deepseek-chat',
    messages: [{ role: 'user', content: '{{user_input}}' }],
    temperature: 0.2,
    stream: false,
  },
  null,
  2,
);

const defaultMapping = {
  response_path: 'choices.0.message.content',
  retrieved_contexts_path: '',
  retrieved_context_ids_path: '',
  tool_calls_path: '',
};

const JsonTextArea: React.FC<React.ComponentProps<typeof TextArea>> = ({ className, ...props }) => (
  <TextArea
    {...props}
    className={['json-textarea', className].filter(Boolean).join(' ')}
    spellCheck={false}
  />
);

const SectionTitle: React.FC<{ title: string; description?: string }> = ({ title, description }) => (
  <div className="eval-form-section-title">
    <Text strong>{title}</Text>
    {description && <Text type="secondary">{description}</Text>}
  </div>
);

const validateJsonText = (_: unknown, value?: string) => {
  if (!value?.trim()) return Promise.resolve();
  try {
    JSON.parse(value);
    return Promise.resolve();
  } catch {
    return Promise.reject(new Error('请输入合法 JSON 对象'));
  }
};

const renderJsonBlock = (value: unknown) => (
  <pre className="result-json-block">
    {typeof value === 'string' ? value : JSON.stringify(value ?? {}, null, 2)}
  </pre>
);

interface EndpointTargetModalProps {
  open: boolean;
  target?: EndpointTarget | null;
  onCancel: () => void;
  onSaved: (target: EndpointTarget) => void;
}

const EndpointTargetModal: React.FC<EndpointTargetModalProps> = ({ open, target, onCancel, onSaved }) => {
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (target) {
      form.setFieldsValue({
        ...target,
        authorization: target.authorization || '',
        response_mapping: { ...defaultMapping, ...(target.response_mapping || {}) },
      });
    } else {
      form.resetFields();
      form.setFieldsValue({
        name: 'DeepSeek Chat',
        endpoint_url: 'https://api.deepseek.com/v1/chat/completions',
        transport_mode: 'json',
        authorization: 'Bearer sk-kkkk',
        extra_headers: '{}',
        request_body_template: DEFAULT_BODY,
        response_mapping: defaultMapping,
        default_test_input: DEFAULT_TEST_INPUT,
      });
    }
  }, [form, open, target]);

  const submit = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      const payload = { ...values, response_mapping: values.response_mapping || {} };
      const saved = target
        ? await api.updateEndpointTarget(target.id, payload)
        : await api.createEndpointTarget(payload);
      message.success(target ? '被测接口已更新' : '被测接口已创建');
      onSaved(saved);
      onCancel();
    } catch (error: any) {
      if (!error?.errorFields) message.error(target ? '更新被测接口失败' : '创建被测接口失败');
    } finally {
      setSubmitting(false);
    }
  };

  const testDraft = async () => {
    try {
      const values = await form.validateFields();
      setTesting(true);
      const result = await api.testEvaluationEndpoint({
        target_config: {
          endpoint_url: values.endpoint_url,
          transport_mode: values.transport_mode || 'json',
          authorization: values.authorization || '',
          extra_headers: values.extra_headers || '{}',
          request_body_template: values.request_body_template,
        },
        response_mapping: values.response_mapping || {},
        row_data: {
          user_input: values.default_test_input || DEFAULT_TEST_INPUT,
        },
      });

      if (result.success) {
        Modal.success({
          width: 860,
          title: '接口试跑成功',
          content: (
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <Descriptions size="small" column={2}>
                <Descriptions.Item label="HTTP 状态码">{result.status_code || '-'}</Descriptions.Item>
                <Descriptions.Item label="耗时">{result.latency_ms || '-'} ms</Descriptions.Item>
              </Descriptions>
              <Text strong>请求体</Text>
              {renderJsonBlock(result.request_body)}
              <Text strong>解析字段</Text>
              {renderJsonBlock(result.extracted_fields || {})}
              {result.mapping_errors && Object.keys(result.mapping_errors).length > 0 && (
                <Alert type="warning" showIcon message="字段映射提示" description={JSON.stringify(result.mapping_errors)} />
              )}
            </Space>
          ),
        });
      } else {
        message.error(`接口试跑失败: ${result.message}`);
      }
    } catch (error: any) {
      if (!error?.errorFields) message.error('接口试跑请求失败');
    } finally {
      setTesting(false);
    }
  };

  return (
    <Modal
      open={open}
      title={target ? '编辑被测接口' : '新建被测接口'}
      width={1120}
      onCancel={onCancel}
      destroyOnClose
      className="endpoint-target-modal"
      footer={[
        <Button key="test" icon={<ExperimentOutlined />} loading={testing} onClick={testDraft}>
          试跑当前配置
        </Button>,
        <Button key="cancel" onClick={onCancel}>
          取消
        </Button>,
        <Button key="submit" type="primary" loading={submitting} onClick={submit}>
          {target ? '保存修改' : '创建接口'}
        </Button>,
      ]}
    >
      <Form form={form} layout="vertical" className="endpoint-target-form">
        <SectionTitle title="接口信息" description="用于评测执行页复用，任务创建时会冻结为快照。" />
        <Row gutter={16}>
          <Col span={5}>
            <Form.Item name="name" label="接口名称" rules={[{ required: true, message: '请输入接口名称' }]}>
              <Input placeholder="DeepSeek Chat" />
            </Form.Item>
          </Col>
          <Col span={11}>
            <Form.Item name="endpoint_url" label="接口地址" rules={[{ required: true, message: '请输入接口地址' }]}>
              <Input placeholder="https://api.deepseek.com/v1/chat/completions" />
            </Form.Item>
          </Col>
          <Col span={3}>
            <Form.Item name="transport_mode" label="返回方式" rules={[{ required: true }]}>
              <Select options={[{ label: 'JSON', value: 'json' }, { label: 'SSE 流式', value: 'sse' }]} />
            </Form.Item>
          </Col>
          <Col span={5}>
            <Form.Item name="authorization" label="Authorization">
              <Input placeholder="Bearer sk-..." />
            </Form.Item>
          </Col>
          <Col span={24}>
            <Form.Item name="description" label="说明">
              <Input placeholder="例如：DeepSeek OpenAI 兼容 Chat Completions 接口" />
            </Form.Item>
          </Col>
        </Row>

        <SectionTitle title="请求模板" description="请求体和 Headers 必须是单个 JSON 对象，可使用 {{user_input}} 或 {{row_data.xxx}}。" />
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="extra_headers" label="附加 Headers(JSON)" rules={[{ validator: validateJsonText }]}>
              <JsonTextArea autoSize={{ minRows: 8, maxRows: 14 }} placeholder={'{\n  "X-App-Id": "demo"\n}'} />
            </Form.Item>
          </Col>
          <Col span={10}>
            <Form.Item
              name="request_body_template"
              label="请求体模板(JSON)"
              rules={[
                { required: true, message: '请填写请求体模板' },
                { validator: validateJsonText },
              ]}
            >
              <JsonTextArea autoSize={{ minRows: 8, maxRows: 14 }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name="default_test_input" label="默认试跑输入">
              <TextArea autoSize={{ minRows: 8, maxRows: 14 }} placeholder="输入一条用于连通性测试的问题" />
            </Form.Item>
          </Col>
        </Row>

        <Divider orientation="left" style={{ margin: '4px 0 16px' }}>响应字段映射</Divider>
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item name={['response_mapping', 'response_path']} label="回答字段">
              <Input placeholder="choices.0.message.content" />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name={['response_mapping', 'retrieved_contexts_path']} label="上下文字段">
              <Input placeholder="data.contexts" />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name={['response_mapping', 'retrieved_context_ids_path']} label="检索ID字段">
              <Input placeholder="data.context_ids" />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name={['response_mapping', 'tool_calls_path']} label="工具调用字段">
              <Input placeholder="trace.tool_calls" />
            </Form.Item>
          </Col>
        </Row>
        <Alert
          type="info"
          showIcon
          message="字段映射使用点路径，数组下标用数字"
          description="例如 OpenAI 兼容接口的回答字段是 choices.0.message.content；httpbin 回显请求体时可以填 json.question。"
        />
      </Form>
    </Modal>
  );
};

export default EndpointTargetModal;

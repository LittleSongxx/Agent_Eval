import React, { useEffect, useState } from 'react';
import { Form, Input, Modal, Select, Space, message } from 'antd';
import type { EndpointTarget } from '../../types';
import * as api from '../../services/api';

const { TextArea } = Input;

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

interface EndpointTargetModalProps {
  open: boolean;
  target?: EndpointTarget | null;
  onCancel: () => void;
  onSaved: (target: EndpointTarget) => void;
}

const EndpointTargetModal: React.FC<EndpointTargetModalProps> = ({ open, target, onCancel, onSaved }) => {
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

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

  return (
    <Modal
      open={open}
      title={target ? '编辑被测接口' : '新建被测接口'}
      width={980}
      onCancel={onCancel}
      onOk={submit}
      confirmLoading={submitting}
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        <Space align="start" style={{ width: '100%' }} size={16}>
          <Form.Item name="name" label="接口名称" rules={[{ required: true }]} style={{ width: 240 }}>
            <Input placeholder="DeepSeek Chat" />
          </Form.Item>
          <Form.Item name="endpoint_url" label="接口地址" rules={[{ required: true }]} style={{ flex: 1 }}>
            <Input placeholder="https://api.deepseek.com/v1/chat/completions" />
          </Form.Item>
          <Form.Item name="transport_mode" label="返回方式" rules={[{ required: true }]} style={{ width: 140 }}>
            <Select options={[{ label: '普通 JSON', value: 'json' }, { label: 'SSE 流式', value: 'sse' }]} />
          </Form.Item>
        </Space>
        <Form.Item name="description" label="说明">
          <Input placeholder="例如：DeepSeek OpenAI 兼容 Chat Completions 接口" />
        </Form.Item>
        <Form.Item name="authorization" label="Authorization">
          <Input placeholder="Bearer sk-..." />
        </Form.Item>
        <Space align="start" style={{ width: '100%' }} size={16}>
          <Form.Item name="extra_headers" label="附加 Headers(JSON)" style={{ width: 360 }}>
            <TextArea autoSize={{ minRows: 5, maxRows: 10 }} />
          </Form.Item>
          <Form.Item name="request_body_template" label="请求体模板(JSON)" rules={[{ required: true }]} style={{ flex: 1 }}>
            <TextArea autoSize={{ minRows: 5, maxRows: 10 }} />
          </Form.Item>
        </Space>
        <Space align="start" style={{ width: '100%' }} size={16}>
          <Form.Item name={['response_mapping', 'response_path']} label="回答字段" style={{ width: 230 }}>
            <Input placeholder="choices.0.message.content" />
          </Form.Item>
          <Form.Item name={['response_mapping', 'retrieved_contexts_path']} label="上下文字段" style={{ width: 230 }}>
            <Input placeholder="data.contexts" />
          </Form.Item>
          <Form.Item name={['response_mapping', 'retrieved_context_ids_path']} label="检索ID字段" style={{ width: 230 }}>
            <Input placeholder="data.context_ids" />
          </Form.Item>
          <Form.Item name={['response_mapping', 'tool_calls_path']} label="工具调用字段" style={{ width: 230 }}>
            <Input placeholder="trace.tool_calls" />
          </Form.Item>
        </Space>
        <Form.Item name="default_test_input" label="默认试跑输入">
          <TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default EndpointTargetModal;

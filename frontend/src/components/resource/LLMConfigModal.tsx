import React, { useEffect, useState } from 'react';
import { Button, Form, Input, InputNumber, Modal, Switch, message } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';
import type { LLMConfig, LLMConfigCreate } from '../../types';
import * as api from '../../services/api';

interface LLMConfigModalProps {
  open: boolean;
  config?: LLMConfig | null;
  onCancel: () => void;
  onSaved: (config: LLMConfig) => void;
}

const LLMConfigModal: React.FC<LLMConfigModalProps> = ({ open, config, onCancel, onSaved }) => {
  const [form] = Form.useForm<LLMConfigCreate>();
  const [submitting, setSubmitting] = useState(false);
  const [testingDraft, setTestingDraft] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (config) {
      form.setFieldsValue({
        name: config.name,
        provider: config.provider,
        api_base_url: config.api_base_url,
        api_key: '',
        model_name: config.model_name,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        is_default: config.is_default,
      });
    } else {
      form.resetFields();
      form.setFieldsValue({ temperature: 0.7, max_tokens: 4096, is_default: false });
    }
  }, [config, form, open]);

  const submit = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      const saved = config
        ? await api.updateLLMConfig(config.id, values.api_key ? values : (({ api_key, ...rest }) => rest)(values))
        : await api.createLLMConfig(values);
      message.success(config ? '更新成功' : '创建成功');
      onSaved(saved);
      onCancel();
    } catch (error: any) {
      if (!error?.errorFields) message.error(config ? '更新失败' : '创建失败');
    } finally {
      setSubmitting(false);
    }
  };

  const testDraft = async () => {
    try {
      const values = await form.validateFields(['name', 'api_base_url', 'api_key', 'model_name']);
      setTestingDraft(true);
      const payload = {
        provider: form.getFieldValue('provider') || 'openai',
        temperature: form.getFieldValue('temperature') ?? 0.01,
        max_tokens: form.getFieldValue('max_tokens') ?? 1024,
        is_default: form.getFieldValue('is_default') ?? false,
        ...values,
      };
      const result = await api.testLLMConfigDraft(payload);
      if (result.success) {
        Modal.success({
          title: '测试成功',
          content: (
            <div>
              <div>延迟：{result.latency_ms ? `${result.latency_ms}ms` : '-'}</div>
              {result.sample_output && (
                <pre style={{ marginTop: 12, maxHeight: 220, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                  {result.sample_output}
                </pre>
              )}
            </div>
          ),
        });
      } else {
        message.error(`连接测试失败: ${result.message}`);
      }
    } catch {
      message.error('请先补全必填配置后再测试');
    } finally {
      setTestingDraft(false);
    }
  };

  return (
    <Modal
      title={config ? '编辑 LLM 配置' : '新增 LLM 配置'}
      open={open}
      onCancel={onCancel}
      width={560}
      destroyOnClose
      footer={[
        <Button key="test" icon={<ExperimentOutlined />} loading={testingDraft} onClick={testDraft}>
          测试当前配置
        </Button>,
        <Button key="cancel" onClick={onCancel}>取消</Button>,
        <Button key="submit" type="primary" loading={submitting} onClick={submit}>
          {config ? '保存' : '创建'}
        </Button>,
      ]}
    >
      <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
        <Form.Item name="name" label="配置名称" rules={[{ required: true, message: '请输入名称' }]}>
          <Input placeholder="例如: GPT-4o 生产环境" />
        </Form.Item>
        <Form.Item name="provider" label="提供商">
          <Input placeholder="例如: openai, anthropic, minimax" />
        </Form.Item>
        <Form.Item name="api_base_url" label="API 地址" rules={[{ required: true, message: '请输入 API 地址' }]}>
          <Input placeholder="https://api.openai.com/v1" />
        </Form.Item>
        <Form.Item name="api_key" label={config ? 'API Key（留空则不修改）' : 'API Key'} rules={config ? [] : [{ required: true, message: '请输入 API Key' }]}>
          <Input.Password placeholder={config ? '留空保持不变' : 'sk-...'} />
        </Form.Item>
        <Form.Item name="model_name" label="模型名称" rules={[{ required: true, message: '请输入模型名称' }]}>
          <Input placeholder="例如: gpt-4o" />
        </Form.Item>
        <Form.Item name="temperature" label="温度 (Temperature)">
          <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item name="max_tokens" label="最大 Token 数">
          <InputNumber min={1} max={128000} step={256} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item name="is_default" label="设为默认" valuePropName="checked">
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default LLMConfigModal;

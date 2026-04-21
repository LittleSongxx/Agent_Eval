import React, { useState } from 'react';
import { Modal, Upload, Typography, message, Space } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import * as api from '../services/api';

const { Text } = Typography;
const { Dragger } = Upload;

interface DataImportModalProps {
  open: boolean;
  datasetId: number;
  onClose: () => void;
  onSuccess: () => void;
}

const DataImportModal: React.FC<DataImportModalProps> = ({
  open,
  datasetId,
  onClose,
  onSuccess,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [resultCount, setResultCount] = useState<number | null>(null);

  const handleUpload = async () => {
    if (!file) {
      message.error('请先选择文件');
      return;
    }
    setUploading(true);
    try {
      const result = await api.importDataset(datasetId, file);
      const count = result.imported_count ?? result.count ?? 0;
      setResultCount(count);
      message.success(`成功导入 ${count} 条数据`);
      onSuccess();
    } catch {
      message.error('导入失败，请检查文件格式');
    } finally {
      setUploading(false);
    }
  };

  const handleClose = () => {
    setFile(null);
    setResultCount(null);
    onClose();
  };

  return (
    <Modal
      title="导入数据"
      open={open}
      onOk={handleUpload}
      onCancel={handleClose}
      okText="确认导入"
      cancelText="取消"
      confirmLoading={uploading}
      okButtonProps={{ disabled: !file }}
    >
      <Dragger
        accept=".csv,.json"
        multiple={false}
        showUploadList={false}
        beforeUpload={(f) => {
          setFile(f);
          setResultCount(null);
          return false;
        }}
        style={{ marginBottom: 16 }}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">点击或拖拽文件到此区域</p>
        <p className="ant-upload-hint">支持 .csv 和 .json 文件</p>
      </Dragger>

      {file && (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>
            已选择文件: <Text strong>{file.name}</Text> ({(file.size / 1024).toFixed(1)} KB)
          </Text>
        </Space>
      )}

      {resultCount !== null && (
        <div style={{ marginTop: 12 }}>
          <Text type="success">已成功导入 {resultCount} 条数据</Text>
        </div>
      )}
    </Modal>
  );
};

export default DataImportModal;

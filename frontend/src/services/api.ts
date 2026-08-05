import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

// ======================== LLM Config ========================

export const listLLMConfigs = () => api.get('/llm-configs').then(r => r.data);
export const createLLMConfig = (data: any) => api.post('/llm-configs', data).then(r => r.data);
export const updateLLMConfig = (id: number, data: any) => api.put(`/llm-configs/${id}`, data).then(r => r.data);
export const deleteLLMConfig = (id: number) => api.delete(`/llm-configs/${id}`);
export const testLLMConfig = (id: number) => api.post(`/llm-configs/${id}/test`).then(r => r.data);
export const testLLMConfigDraft = (data: any) => api.post('/llm-configs/test-draft', data).then(r => r.data);

// ======================== Dataset ========================

export const listDatasets = () => api.get('/datasets').then(r => r.data);
export const createDataset = (data: any) => api.post('/datasets', data).then(r => r.data);
export const getDataset = (id: number) => api.get(`/datasets/${id}`).then(r => r.data);
export const deleteDataset = (id: number) => api.delete(`/datasets/${id}`);
export const listDatasetRows = (id: number, page = 1, pageSize = 20) =>
  api.get(`/datasets/${id}/rows`, { params: { page, page_size: pageSize } }).then(r => r.data);
export const createDatasetRow = (id: number, data: any) =>
  api.post(`/datasets/${id}/rows`, data).then(r => r.data);
export const deleteDatasetRow = (id: number, rowId: number) =>
  api.delete(`/datasets/${id}/rows/${rowId}`);
export const importDataset = (id: number, file: File) => {
  const fd = new FormData();
  fd.append('file', file);
  return api.post(`/datasets/${id}/import`, fd).then(r => r.data);
};
export const exportDataset = (id: number, format: 'csv' | 'json') =>
  api.get(`/datasets/${id}/export`, { params: { format }, responseType: 'blob' }).then(r => ({
    blob: r.data as Blob,
    contentDisposition: r.headers['content-disposition'] as string | undefined,
    contentType: r.headers['content-type'] as string | undefined,
  }));

// ======================== RAG Dataset Jobs ========================

export const listRagDatasetJobs = () => api.get('/rag-dataset-jobs').then(r => r.data);
export const createRagDatasetJob = (data: any) => api.post('/rag-dataset-jobs', data).then(r => r.data);
export const getRagDatasetJob = (id: number) => api.get(`/rag-dataset-jobs/${id}`).then(r => r.data);
export const uploadRagDatasetJobDocuments = (id: number, files: File[]) => {
  const fd = new FormData();
  files.forEach((file) => fd.append('files', file));
  return api.post(`/rag-dataset-jobs/${id}/documents`, fd).then(r => r.data);
};
export const listRagDatasetJobChunks = (id: number, documentId?: number) =>
  api.get(`/rag-dataset-jobs/${id}/chunks`, { params: documentId ? { document_id: documentId } : undefined }).then(r => r.data);
export const listRagDatasetJobSamples = (id: number, page = 1, pageSize = 20, status?: string) =>
  api.get(`/rag-dataset-jobs/${id}/samples`, { params: { page, page_size: pageSize, status } }).then(r => r.data);
export const startRagDatasetJob = (id: number) => api.post(`/rag-dataset-jobs/${id}/start`).then(r => r.data);
export const retryFailedRagDatasetJob = (id: number) => api.post(`/rag-dataset-jobs/${id}/retry-failed`).then(r => r.data);

// ======================== Metric ========================

export const listMetrics = async () => {
  const first = await api.get('/metrics', { params: { page: 1, page_size: 100 } }).then(r => r.data);
  if (Array.isArray(first)) return first;

  const firstItems = first?.items || [];
  const total = Number(first?.total || firstItems.length);
  const pageSize = Number(first?.page_size || firstItems.length || 100);
  if (!total || firstItems.length >= total || !pageSize) return firstItems;

  const pageCount = Math.ceil(total / pageSize);
  const restPages = await Promise.all(
    Array.from({ length: pageCount - 1 }, (_, index) =>
      api.get('/metrics', { params: { page: index + 2, page_size: pageSize } }).then(r => r.data)
    )
  );
  return [
    ...firstItems,
    ...restPages.flatMap((page) => (Array.isArray(page) ? page : page?.items || [])),
  ];
};
export const createMetric = (data: any) => api.post('/metrics', data).then(r => r.data);
export const updateMetric = (id: number, data: any) => api.put(`/metrics/${id}`, data).then(r => r.data);
export const deleteMetric = (id: number) => api.delete(`/metrics/${id}`);

// ======================== Endpoint Targets ========================

export const listEndpointTargets = () => api.get('/endpoint-targets').then(r => r.data);
export const createEndpointTarget = (data: any) => api.post('/endpoint-targets', data).then(r => r.data);
export const updateEndpointTarget = (id: number, data: any) => api.put(`/endpoint-targets/${id}`, data).then(r => r.data);
export const deleteEndpointTarget = (id: number) => api.delete(`/endpoint-targets/${id}`);
export const testEndpointTarget = (id: number, data: any) => api.post(`/endpoint-targets/${id}/test`, data).then(r => r.data);

// ======================== Scenario ========================

export const listScenarios = () => api.get('/scenarios').then(r => r.data);
export const createScenario = (data: any) => api.post('/scenarios', data).then(r => r.data);
export const updateScenario = (id: number, data: any) => api.put(`/scenarios/${id}`, data).then(r => r.data);
export const getScenario = (id: number) => api.get(`/scenarios/${id}`).then(r => r.data);
export const deleteScenario = (id: number) => api.delete(`/scenarios/${id}`);
export const getPresetScenarios = () => api.get('/scenarios/presets').then(r => r.data);

// ======================== Evaluation ========================

export const listEvaluations = () => api.get('/evaluations').then(r => r.data);
export const createEvaluation = (data: any) => api.post('/evaluations', data).then(r => r.data);
export const debugEvaluation = (data: any) => api.post('/evaluations/debug', data).then(r => r.data);
export const getEvaluation = (id: number) => api.get(`/evaluations/${id}`).then(r => r.data);
export const getEvaluationLogs = (id: number) => api.get(`/evaluations/${id}/logs`).then(r => r.data);
export const cancelEvaluation = (id: number) => api.post(`/evaluations/${id}/cancel`);
export const testEvaluationEndpoint = (data: any) => api.post('/evaluations/test-endpoint', data).then(r => r.data);

// WebSocket 实时进度：走 vite /ws 代理到后端，断开时前端会回退到轮询
export const evaluationWsUrl = (taskId: number) => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/evaluations/${taskId}`;
};

// ======================== Report ========================

export const listReports = (params?: any) =>
  api.get('/reports', { params }).then(r => r.data);
export const getReportSummary = (evalId: number) =>
  api.get(`/reports/${evalId}/summary`).then(r => r.data);
export const getReportRows = (evalId: number, page = 1, pageSize = 20, status?: string) =>
  api.get(`/reports/${evalId}/rows`, { params: { page, page_size: pageSize, status } }).then(r => r.data);
export const getReportRowDetail = (evalId: number, rowId: number) =>
  api.get(`/reports/${evalId}/rows/${rowId}`).then(r => r.data);
export const updateReportRowReview = (evalId: number, rowId: number, data: any) =>
  api.patch(`/reports/${evalId}/rows/${rowId}/review`, data).then(r => r.data);
export const compareReport = (evalId: number, baselineEvalId: number) =>
  api.get(`/reports/${evalId}/compare`, { params: { baseline_eval_id: baselineEvalId } }).then(r => r.data);

// ======================== Blind Test ========================

export const listBlindTests = () => api.get('/blind-tests').then(r => r.data);
export const createBlindTest = (data: any) => api.post('/blind-tests', data).then(r => r.data);
export const getBlindTest = (id: number) => api.get(`/blind-tests/${id}`).then(r => r.data);
export const getBlindTestSummary = (id: number) => api.get(`/blind-tests/${id}/summary`).then(r => r.data);
export const getBlindTestRows = (id: number, page = 1, pageSize = 20, status?: string) =>
  api.get(`/blind-tests/${id}/rows`, { params: { page, page_size: pageSize, status } }).then(r => r.data);
export const voteBlindTestRow = (id: number, rowId: number, data: any) =>
  api.post(`/blind-tests/${id}/rows/${rowId}/vote`, data).then(r => r.data);
export const cancelBlindTest = (id: number) => api.post(`/blind-tests/${id}/cancel`).then(r => r.data);
export const testBlindTestTarget = (data: any) => api.post('/blind-tests/test-target', data).then(r => r.data);

export default api;

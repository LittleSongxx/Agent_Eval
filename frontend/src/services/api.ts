import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

// ======================== LLM Config ========================

export const listLLMConfigs = () => api.get('/llm-configs').then(r => r.data);
export const createLLMConfig = (data: any) => api.post('/llm-configs', data).then(r => r.data);
export const updateLLMConfig = (id: number, data: any) => api.put(`/llm-configs/${id}`, data).then(r => r.data);
export const deleteLLMConfig = (id: number) => api.delete(`/llm-configs/${id}`);
export const testLLMConfig = (id: number) => api.post(`/llm-configs/${id}/test`).then(r => r.data);

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

// ======================== Metric ========================

export const listMetrics = () => api.get('/metrics').then(r => r.data);
export const createMetric = (data: any) => api.post('/metrics', data).then(r => r.data);
export const deleteMetric = (id: number) => api.delete(`/metrics/${id}`);

// ======================== Scenario ========================

export const listScenarios = () => api.get('/scenarios').then(r => r.data);
export const createScenario = (data: any) => api.post('/scenarios', data).then(r => r.data);
export const getScenario = (id: number) => api.get(`/scenarios/${id}`).then(r => r.data);
export const deleteScenario = (id: number) => api.delete(`/scenarios/${id}`);
export const getPresetScenarios = () => api.get('/scenarios/presets').then(r => r.data);

// ======================== Evaluation ========================

export const listEvaluations = () => api.get('/evaluations').then(r => r.data);
export const createEvaluation = (data: any) => api.post('/evaluations', data).then(r => r.data);
export const getEvaluation = (id: number) => api.get(`/evaluations/${id}`).then(r => r.data);
export const cancelEvaluation = (id: number) => api.post(`/evaluations/${id}/cancel`);

// ======================== Report ========================

export const getReportSummary = (evalId: number) =>
  api.get(`/reports/${evalId}/summary`).then(r => r.data);
export const getReportRows = (evalId: number, page = 1, pageSize = 20, status?: string) =>
  api.get(`/reports/${evalId}/rows`, { params: { page, page_size: pageSize, status } }).then(r => r.data);
export const getReportRowDetail = (evalId: number, rowId: number) =>
  api.get(`/reports/${evalId}/rows/${rowId}`).then(r => r.data);

export default api;

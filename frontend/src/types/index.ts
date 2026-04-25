export interface LLMConfig {
  id: number;
  name: string;
  provider: string;
  api_base_url: string;
  api_key_masked: string;
  model_name: string;
  temperature: number;
  max_tokens: number;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface LLMConfigCreate {
  name: string;
  provider?: string;
  api_base_url: string;
  api_key: string;
  model_name: string;
  temperature?: number;
  max_tokens?: number;
  is_default?: boolean;
}

export interface LLMTestResult {
  success: boolean;
  message: string;
  latency_ms?: number;
}

export interface FieldDefinition {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

export interface Dataset {
  id: number;
  name: string;
  description: string;
  sample_type: string;
  field_schema: FieldDefinition[];
  row_count: number;
  created_at: string;
  updated_at: string;
}

export interface DatasetRow {
  id: number;
  dataset_id: number;
  row_index: number;
  data: Record<string, any>;
  created_at: string;
}

export interface MetricDefinition {
  id: number;
  name: string;
  display_name: string;
  metric_type: string;
  config: Record<string, any>;
  category: string;
  is_builtin: boolean;
  created_at: string;
}

export interface ScenarioMetric {
  id: number;
  metric_definition_id: number;
  weight: number;
  pass_threshold: number | null;
  metric_definition?: MetricDefinition;
}

export interface EvalScenario {
  id: number;
  name: string;
  description: string;
  scene_type: string;
  sample_type: string;
  is_preset: boolean;
  metrics: ScenarioMetric[];
  created_at: string;
}

export interface EvalTask {
  id: number;
  name: string;
  dataset_id: number;
  scenario_id: number;
  llm_config_id: number;
  status: string;
  progress: number;
  total_rows: number;
  completed_rows: number;
  error_message: string | null;
  summary_scores: Record<string, any> | null;
  scenario_snapshot?: Record<string, any> | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  dataset?: Dataset;
  scenario?: EvalScenario;
  llm_config?: LLMConfig;
}

export interface EvalRowResult {
  id: number;
  eval_task_id: number;
  row_index: number;
  metric_scores: Record<string, { score: number | null; reason: string }>;
  is_pass: boolean | null;
  execution_time_ms: number | null;
  error: string | null;
  dataset_row?: DatasetRow;
  created_at: string;
}

export interface ReportSummary {
  eval_task: EvalTask;
  total_count: number;
  pass_count: number;
  fail_count: number;
  error_count: number;
  pass_rate: number;
  metric_summary: Record<string, { mean: number; min: number; max: number; pass_rate: number }>;
}

export interface PaginatedResponse<T> {
  total: number;
  page: number;
  page_size: number;
  items: T[];
}

export interface RagDatasetDocument {
  id: number;
  job_id: number;
  filename: string;
  file_path?: string | null;
  char_count: number;
  chunk_count: number;
  created_at: string;
}

export interface RagDatasetChunk {
  id: number;
  document_id: number;
  chunk_index: number;
  chunk_key: string;
  content: string;
  char_count: number;
  suggested_question_count: number;
  allocated_question_count: number;
  quality_label: 'good' | 'medium' | 'low' | 'filtered';
  quality_score: number;
  quality_reasons: string[];
  generation_status: string;
  generation_error?: string | null;
  created_at: string;
}

export interface RagDatasetSample {
  id: number;
  job_id: number;
  document_id: number;
  chunk_id?: number | null;
  dataset_row_id?: number | null;
  question: string;
  reference: string;
  reference_context_ids: string[];
  source_chunk_ids: string[];
  response?: string | null;
  retrieved_contexts?: string[] | null;
  retrieved_context_ids?: string[] | null;
  status: string;
  error_message?: string | null;
  retry_count: number;
  selected: boolean;
  created_at: string;
  updated_at?: string | null;
}

export interface RagDatasetJob {
  id: number;
  name: string;
  description?: string | null;
  status: string;
  target_endpoint_url?: string | null;
  target_transport_mode: 'json' | 'sse';
  target_authorization_masked: string;
  target_extra_headers: string;
  target_request_body_template: string;
  target_response_mode: 'answer_only' | 'answer_with_contexts';
  target_system_prompt?: string | null;
  question_count_mode: 'auto' | 'custom';
  requested_question_count?: number | null;
  suggested_question_count?: number | null;
  total_documents: number;
  total_chunks: number;
  total_samples: number;
  completed_samples: number;
  failed_samples: number;
  error_message?: string | null;
  logs: string;
  started_at?: string | null;
  finished_at?: string | null;
  created_at: string;
  updated_at?: string | null;
  dataset_id?: number | null;
  dataset?: Dataset | null;
  question_llm_config?: LLMConfig | null;
  target_llm_config?: LLMConfig | null;
  documents: RagDatasetDocument[];
  generation_summary?: {
    supported_metrics: string[];
    unsupported_metrics: string[];
    notes: string[];
  };
}

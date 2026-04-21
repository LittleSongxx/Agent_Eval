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

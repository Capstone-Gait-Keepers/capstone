import { reactive } from 'vue'
import type { Metrics, User } from '@/types'
import { getMetrics } from '@/backend_interface';

const store = reactive<{
  user: User | null;
  connected_users: string[];
  view_sections: Record<string, boolean>;
  data: Metrics | null;
}>({
  user: {
    name: "Marjorie",
    email: "marjorie.smith@gmail.com",
    password: "thisisapassword",
    sensor_id: 39,
  }, // TODO: Default should be null
  connected_users: ['dan@raymond.ch'],
  view_sections: {
    Balance: true,
    Neurodegenerative: true,
    Dementia: true,
  },
  data: null,
});

export const metric_sections: Record<string, string[]> = {
  "Balance": ['var_coef', 'stga'],
  "Neurodegenerative": ['phase_sync', 'cond_entropy', 'stga'],
  "Dementia": ['stride_time', 'cadence', 'var_coef', 'stga'],
};

export function validSection(section: string): boolean {
  if (store.data?.metrics === undefined) {
    return false;
  }
  const metrics = metric_sections[section];
  return store.view_sections[section] && metrics.some(validMetric);
}

export function validMetric(metric: string): boolean {
  const metric_data = store.data?.metrics[metric];
  if (metric_data === undefined) {
    return false;
  }
  return metric_data.some((val) => val !== null);
}

export const fetchData = async () => {
  if (store.data === null) {
    const resp = await getMetrics();
    if (resp !== null)
      store.data = resp;
    else
      console.error('Failed to get metrics');
  }
}

export default store;
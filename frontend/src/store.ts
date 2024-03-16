import { reactive } from 'vue'
import type { Metrics, User } from '@/types'
import { getMetrics } from '@/backend_interface';

export enum Section {
  Balance = 'Balance',
  Neurodegenerative = 'Neurodegenerative',
  Dementia = 'Dementia',
}

export const store = reactive<{
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
    "Balance": true,
    "Neurodegenerative": true,
    "Dementia": true,
  },
  data: null,
});

export const metric_sections: Record<Section, string[]> = {
  "Balance": ['var_coef', 'STGA'],
  "Neurodegenerative": ['phase_sync', 'conditional_entropy', 'STGA'],
  "Dementia": ['stride_time', 'cadence', 'var_coef', 'STGA'],
};

export function validSection(section: Section): boolean {
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
    console.log(resp);
    if (resp !== null)
      store.data = resp;
    else
      console.error('Failed to get metrics');
  }
}

export function datesBackIndex(days: number): number {
  if (store.data === null) {
    return 0;
  }
  // TODO: Current date as a reference?
  const date = new Date(store.data.dates[store.data.dates.length - 1]);
  date.setDate(date.getDate() - days);
  return -store.data.dates.filter((d) => new Date(d) >= date).length;
}

export function getMetric(metric: string, days: number): number[] {
  if (store.data === null) {
    return [];
  }
  return store.data.metrics[metric].slice(datesBackIndex(days));
}

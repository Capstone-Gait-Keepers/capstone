import { reactive } from 'vue'
import type { Metrics, User } from '@/types'
import { getMetrics } from '@/backend_interface';

export enum Section {
  balance = 'balance',
  neurodegenerative = 'neurodegenerative',
  dementia = 'dementia',
  reliability = 'reliability',
}

export const section_titles: Record<Section, string> = {
  [Section.balance]: 'Balance Indicators',
  [Section.neurodegenerative]: 'Neurodegenerative Indicators',
  [Section.dementia]: 'Dementia Indicators',
  [Section.reliability]: 'System Reliability',
};

export const store = reactive<{
  user: User | null;
  connected_users: string[];
  view_sections: Record<Section, boolean>;
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
    [Section.balance]: true,
    [Section.neurodegenerative]: true,
    [Section.dementia]: true,
    [Section.reliability]: true,
  },
  data: null,
});

export const metric_sections: Record<Section, string[]> = {
  [Section.balance]: ['var_coef', 'STGA'],
  [Section.neurodegenerative]: ['phase_sync', 'conditional_entropy', 'STGA'],
  [Section.dementia]: ['stride_time', 'cadence', 'var_coef', 'STGA'],
  [Section.reliability]: ['step_count']
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

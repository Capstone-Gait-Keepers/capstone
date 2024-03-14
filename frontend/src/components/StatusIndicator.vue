<template>
  <div class="status-indicator" v-if="status">
    <b>This {{ timespan }}</b>
    <img v-if="status === Status.Good" src="@/assets/icon-thumbs-up.svg" />
    <div v-else-if="status === Status.Neutral" class="flat-bar"></div>
    <img v-else-if="status === Status.Bad" src="@/assets/icon-warning.svg" />
    <p>2 of 4 metrics showed positive changes compared with last month!</p>
  </div>
</template>

<script setup lang="ts">
import { store, Section, metric_sections, datesBackIndex } from '@/store';
const { data } = store;

enum Status {
  Good = 'good',
  Neutral = 'neutral',
  Bad = 'bad',
}

const { section, timespan } = defineProps<{section: Section, timespan: "Month" | "Year"}>();
let status: Status | null = null;

const metrics = metric_sections[section];
const metric_changes: Record<string, Status> = {};
function getMetricStatus(metric_values: number[]): Status {
  if (metric_values.length === 0) return Status.Neutral;
  const change = metric_values[metric_values.length - 1] - metric_values[0];
  if (change > .3) return Status.Good;
  if (change < -.2) return Status.Bad;
  return Status.Neutral;
}

if (data !== null) {
  for (const metric of metrics) {
    const values = data.metrics[metric];
    const values_in_period = values.slice(datesBackIndex(timespan === "Month" ? 30 : 365));
    metric_changes[metric] = getMetricStatus(values_in_period);
  }
  console.log(metric_changes);
  const changes = Object.values(metric_changes);
  if (changes.includes(Status.Bad)) status = Status.Bad;
  else if (changes.includes(Status.Good)) status = Status.Good;
  else status = Status.Neutral;
}
</script>


<style>
.status-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 0 2rem;
}

.status-indicator img {
  width: 4rem;
  height: 4rem;
}

.flat-bar {
  border-bottom: 4px solid var(--color-main);
  width: 80px;
  height: 1.8rem;
  margin: 1rem 0;
}
</style>
<template>
  <div class="status-indicator" v-if="status">
    <b>This {{ timespan }}</b>
    <img v-if="status === Status.Good" src="@/assets/icon-thumbs-up.svg" />
    <div v-else-if="status === Status.Neutral" class="flat-bar"></div>
    <img v-else-if="status === Status.Bad" src="@/assets/icon-warning.svg" />
    <p v-if="status === Status.Good">
      {{ Object.values(metric_changes).filter(a => a === Status.Good).length }} of {{ metrics.length }} metrics showed positive changes compared with last {{ timespan }}!
    </p>
    <p v-else-if="status === Status.Neutral">
      No changes in {{ section }} related metrics this month.
    </p>
    <p v-else-if="status === Status.Bad">
      There were negative changes in {{ Object.values(metric_changes).filter(a => a === Status.Bad).length }}
      of the {{ metrics.length }} metrics related to {{ section }} this month.
    </p>
  </div>
</template>

<script setup lang="ts">
import { store, Section, metric_sections, datesBackIndex, upMeaning, Status } from '@/store';
const { data } = store;


const { section, timespan } = defineProps<{section: Section, timespan: "Month" | "Year"}>();
let status: Status | null = null;

const metrics = metric_sections[section];
const metric_changes: Record<string, Status> = {};

function getMetricStatus(metric: string, days: number): Status {
  if (!data?.metrics[metric])
    return Status.Neutral;
  const metric_values = data.metrics[metric].slice(datesBackIndex(days));
  if (metric_values.length === 0)
    return Status.Neutral;
  const change = metric_values[metric_values.length - 1] - metric_values[0];
  if (change > .3) return upMeaning[metric];
  if (change < -.2) return upMeaning[metric] === Status.Good ? Status.Bad : Status.Good;
  return Status.Neutral;
}

if (data !== null) {
  for (const metric of metrics) {
    metric_changes[metric] = getMetricStatus(metric, timespan === "Month" ? 30 : 365);
  }
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
  margin-top: 1.8rem;
  margin-bottom: 2rem;
}
</style>
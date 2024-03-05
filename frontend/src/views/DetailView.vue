<template>
  <BasePage>
    <h1>Learn More</h1>
    <div v-for="name in metric_names" v-if="metrics !== null" :key="name">
      <h2>{{ name }}</h2>
      <Plot
        :x="getDates(metrics[name])"
        :y="getValues(metrics[name])"
        :xlabel="'Date'"
        :ylabel="name"
        plot_type="scatter"
      />
    </div>
  </BasePage>
</template>


<script setup lang="ts">
import BasePage from '@/components/BasePage.vue';
import Plot from '@/components/Plot.vue';
import { ref, onMounted } from 'vue';
import type { MetricSequence, Metrics } from '@/types';
import { getMetrics } from '@/backend_interface';


const metrics = ref<Metrics | null>(null);
const metric_names = ref<string[]>([]);

onMounted(async () => {
  metrics.value = await getMetrics();
  if (metrics.value === null) {
    console.error('Failed to get metrics');
    return;
  }
  metric_names.value = Object.keys(metrics.value);
  metric_names.value = metric_names.value.filter((name) => name !== 'date');
});

function getDates(seq: MetricSequence): string[] {
  return seq.map((metric) => metric.date);
}

function getValues(seq: MetricSequence): any[] {
  return seq.map((metric) => metric.value);
}
</script>

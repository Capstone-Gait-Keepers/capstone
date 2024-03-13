<template>
  <BasePage>
    <h1>Learn More</h1>
    <Accordion v-model:isOpen="isAccordionOpen" header="Accordion">
    hi
    </Accordion>
    <div v-for="metric_keys, header in metric_categories" v-if="metrics !== null" :key="header" class="category">
      <h2>{{ header }} Metrics</h2>
      <div v-for="key in metric_keys" :key="key" class="metric">
        <h3>{{ metric_titles.get(key) }}</h3>
        <Plot
          :x="getDates(metrics[key])"
          :y="getValues(metrics[key])"
          :xlabel="'Date'"
          :ylabel="metric_titles.get(key)"
          plot_type="scatter"
        />
      </div>
    </div>
  </BasePage>
</template>


<script setup lang="ts">
import BasePage from '@/components/BasePage.vue';
import Plot from '@/components/Plot.vue';
import Accordion from '@/components/Accordion.vue';
import { ref, onMounted } from 'vue';
import type { MetricSequence, Metrics } from '@/types';
import { getMetrics } from '@/backend_interface';


const isAccordionOpen = ref(false);
const metrics = ref<Metrics | null>(null);
const metric_categories = {
  Balance: ['var_coef', 'stga'],
  Neurodegenerative: ['phase_sync', 'cond_entropy', 'stga'],
  Dementia: ['stride_time', 'cadence', 'var_coef', 'stga'],
};


const metric_titles = new Map([
  ['var_coef', 'Stride Time Coefficient of Variation'],
  ['stga', 'Stride Time Gait Asymmetry'],
  ['phase_sync', 'Stride Time Phase Synchronization'],
  ['cond_entropy', 'Stride Time Conditional Entropy'],
  ['stride_time', 'Stride Time'],
  ['cadence', 'Cadence'],
]);

onMounted(async () => {
  metrics.value = await getMetrics();
  if (metrics.value === null) {
    console.error('Failed to get metrics');
    return;
  }
  const given_metrics = new Set(Object.keys(metrics.value));
  if (given_metrics.size !== metric_titles.size) {
    console.error('Given metrics do not match expected metrics');
  }
});

function getDates(seq: MetricSequence): string[] {
  return seq.map((metric) => metric.date);
}

function getValues(seq: MetricSequence): any[] {
  return seq.map((metric) => metric.value);
}
</script>

<style scoped>
.category {
  margin-bottom: 8rem;
}

.metric {
  margin-bottom: 8rem;
}
</style>
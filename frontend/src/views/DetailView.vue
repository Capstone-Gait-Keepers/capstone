<template>
  <BasePage>
    <div class="main">
      <h1>Breakdown</h1>
      <p>Dive into the measurements WalkWise has collected to learn how changes are determined.</p>
      <div v-for="metric_keys, header in metric_categories" v-if="metrics !== null" :key="header" class="category">
        <h2>{{ header }} Metrics</h2>
        <Accordion v-for="key in metric_keys" :key="key" :header="metric_titles.get(key)" class="metric">
          <p>{{ metric_descriptions.get(key) }}</p>
          <Plot
            :x="getDates(metrics[key])"
            :y="getValues(metrics[key])"
            xlabel="Date"
            :ylabel="metric_titles.get(key)"
            plot_type="scatter"
          />
        </Accordion>
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

const metric_descriptions = new Map([
  ["var_coef", "The Stride Time Coefficient of Variation measures the consistency of step timing during walking. A low coefficient indicates consistent step timing, while a high coefficient suggests variability. It helps assess how regular or irregular someone's walking pattern is, which can indicate potential issues with movement or stability."],
  ["stga", "Stride Time Gait Asymmetry refers to differences in the timing of steps between the left and right legs during walking. It highlights imbalance or irregularity in step timing. Detecting such asymmetry helps assess gait health and identify potential issues affecting movement coordination or stability."],
  ["phase_sync", "Stride Time Phase Synchronisation examines how well steps align in time during walking. When steps are synchronized, people walk smoothly. Desynchronization may indicate instability or irregular gait. It helps understand coordination and detect abnormalities in movement patterns."],
  ["cond_entropy", "Stride time conditional entropy measures the consistency or unpredictability of step timing during walking or running. Low entropy indicates consistent timing, while high entropy suggests variability. It helps assess gait stability and detect potential movement abnormalities."],
  ["stride_time", "Stride time refers to the time it takes to complete one full step while walking or running. It's the duration from when your foot touches the ground to when it touches the ground again. Monitoring stride time helps understand the rhythm and pace of movement."],
  ["cadence", "Cadence is the rhythm or pace at which you walk or run, determined by how many steps you take per minute. It's like the beat of a song for your movement. Faster cadence means quicker steps, while slower cadence means slower steps."],
  // Measurements Collected: The number of measurements collected by the sensor each day. This can also be an indicator of daily activity.
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
.main {
  display: grid;
  gap: 1rem;
}

.category {
  margin-bottom: 2rem;
}

.metric {
  margin-top: 2rem;
  margin-bottom: 1rem;
}

.metric p {
  margin: 1rem;
}
</style>

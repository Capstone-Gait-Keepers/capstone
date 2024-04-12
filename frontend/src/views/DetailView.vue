<template>
  <BasePage>
    <div class="main">
      <h1>Breakdown</h1>
      <p>Dive into the measurements WalkWise has collected to learn how changes are determined.</p>
      <span v-if="store.data !== null" v-for="metric_keys, header in metric_sections">
        <div v-if="validSection(header)" :key="header" class="category" :id="header">
          <h2>{{ section_titles[header] }}</h2>
          <span v-for="key in metric_keys" :key="key">
            <Accordion v-if="validMetric(key)" :startOpen="hash === header" class="metric">
              <template v-slot:header>
                <div class="accordion-header">
                  {{ metric_titles[key] }}
                  <p
                    :style="{'color': getMetricColor(key)}">
                    {{ metric_changes[key] > 0 ? '+' : '' }}{{ metric_changes[key] }}%
                  </p>
                </div>
              </template>
              <p>{{ metric_descriptions[key] }}</p>
              <InteractivePlot
                :x="cleanedDates(key)"
                :y="cleanedMetric(key)"
                xlabel="Date"
                :ylabel="metric_units[key]"
                :hline="metric_controls[key]"
                plot_type="scatter"
              />
            </Accordion>
          </span>
        </div>
      </span>
      <ListLoader class="loading pulse" primaryColor="#e3e3e3" v-else /> 
    </div>
  </BasePage>
</template>


<script setup lang="ts">
import { ListLoader } from 'vue-content-loader'

import BasePage from '@/components/BasePage.vue';
import InteractivePlot from '@/components/InteractivePlot.vue';
import Accordion from '@/components/Accordion.vue';
import { store, metric_sections, section_titles, validMetric, validSection } from '@/store';


const hash = window.location.hash.slice(1);

const metric_titles: Record<string, string> = {
  var_coef: 'Stride Time Coefficient of Variation',
  STGA: 'Stride Time Gait Asymmetry',
  phase_sync: 'Stride Time Phase Synchronization',
  conditional_entropy: 'Stride Time Conditional Entropy',
  stride_time: 'Stride Time',
  cadence: 'Cadence',
  step_count: 'Measurements Collected',
};

const metric_units: Record<string, string> = {
  var_coef: 'Coefficient of Variation',
  STGA: 'Asymmetry Index',
  phase_sync: 'Phase Synchronization Index',
  conditional_entropy: 'Conditional Entropy',
  stride_time: 'Seconds',
  cadence: 'Steps per second',
  step_count: 'Steps detected',
};

const metric_descriptions: Record<string, string> = {
  "var_coef": "The Stride Time Coefficient of Variation measures the consistency of step timing during walking. A low coefficient indicates consistent step timing, while a high coefficient suggests variability. It helps assess how regular or irregular someone's walking pattern is, which can indicate potential issues with movement or stability.",
  "STGA": "Stride Time Gait Asymmetry refers to differences in the timing of steps between the left and right legs during walking. It highlights imbalance or irregularity in step timing. Detecting such asymmetry helps assess gait health and identify potential issues affecting movement coordination or stability.",
  "phase_sync": "Stride Time Phase Synchronisation examines how well steps align in time during walking. When steps are synchronized, people walk smoothly. Desynchronization may indicate instability or irregular gait. It helps understand coordination and detect abnormalities in movement patterns.",
  "conditional_entropy": "Stride time conditional entropy measures the consistency or unpredictability of step timing during walking or running. Low entropy indicates consistent timing, while high entropy suggests variability. It helps assess gait stability and detect potential movement abnormalities.",
  "stride_time": "Stride time refers to the time it takes to complete one full step while walking or running. It's the duration from when your foot touches the ground to when it touches the ground again. Monitoring stride time helps understand the rhythm and pace of movement.",
  "cadence": "Cadence is the rhythm or pace at which you walk or run, determined by how many steps you take per minute. It's like the beat of a song for your movement. Faster cadence means quicker steps, while slower cadence means slower steps.",
  "step_count": "The number of measurements collected by the sensor each day. This can also be an indicator of daily activity."
};

const metric_controls: Record<string, number> = {
  "STGA": 0.023,
  "stride_time": 1.036,
  "cadence": 1.692,
  "var_coef": 0.017,
  "phase_sync": 0.812,
  "conditional_entropy": 0.007,
};

const metric_changes: Record<string, number> = {
  "var_coef": 0,
  "STGA": -10,
  "phase_sync": 0,
  "conditional_entropy": 0,
  "stride_time": 0,
  "cadence": 0,
  "step_count": 0,
};

function cleanedMetric(metric: string) {
  return store.data?.metrics[metric].filter((x: number) => x !== null) || [];
}

function getMetricColor(metric: string) {
  if (metric_changes[metric] === 0) {
    return '';
  } else if (metric_changes[metric] > 0) {
    return 'var(--red)';
  } else {
    return 'var(--green)';
  }
}

function cleanedDates(metric: string) {
  // Drop dates where the metric data is null
  const metricData = store.data?.metrics[metric];
  if (metricData === undefined) {
    return [];
  }
  return store.data?.dates.filter((_, i) => metricData[i] !== null) || [];
}
</script>

<style scoped>
.main {
  margin-bottom: 8rem;
}

.main h1 {
  margin-bottom: 1rem;
}

.main p {
  margin-bottom: 2rem;
}

.accordion-header {
  display: flex;
  justify-content: space-between;
  width: 100%;
  align-items: center;
  font-weight: bold;
}

.accordion-header p {
  font-weight: bold;
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

.loading {
  padding: 2rem;
  border-radius: 1rem;
}
</style>

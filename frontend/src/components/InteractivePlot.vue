<template>
  <Plot
    :x="x.slice(slice_index)"
    :y="y.slice(slice_index)"
    :xlabel="xlabel"
    :ylabel="ylabel"
    :hline="hline"
    :plot_type="plot_type"
    :title="title"
  />
  <div class="footer">
    <div>
      <p>Legend</p>
      <div class="legend-entry"><div class="flat-bar"/>You</div>
      <div class="legend-entry"><div class="flat-bar" style="border-color: black; border-style: dashed;"/>Typical</div>
    </div>
    <div class="buttons">
      <button @click.stop="span = 30" :class="{'active': span == 30}">Month</button>
      <button @click.stop="span = 365" :class="{'active': span == 365}">Year</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, withDefaults, computed } from 'vue';
import { datesBackIndex } from '@/store';
import Plot from '@/components/Plot.vue';

const { x } = withDefaults(defineProps<{
  x: Array<string>,
  y: Array<number>,
  xlabel: string,
  ylabel: string,
  plot_type: string,
  hline?: number,
  title?: string,
}>(), {
  plot_type: 'scatter',
});

const span = ref(30);
const slice_index = computed(() => datesBackIndex(span.value));
</script>

<style scoped>
.footer {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
}

.buttons {
  display: flex;
  justify-content: center;
  margin: 1rem;
}

button {
  font-size: .8em;
  color: #525252;
  background-color: #F4F4F4;
  box-shadow: none;
}

.buttons button:first-of-type {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}

.buttons button:last-of-type {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}

.active {
  background-color: var(--color-main);
  color: white;
}

.flat-bar {
  border: 0px;
  border-bottom: 3px solid var(--color-main);
  width: 40px;
  margin: 0rem;
}

.legend-entry {
  display: flex;
  align-items: center;
  gap: .5rem;
}
</style>
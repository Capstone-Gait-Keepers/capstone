<template>
  <Plot
    :x="x.slice(slice_index)"
    :y="y.slice(slice_index)"
    :xlabel="xlabel"
    :ylabel="ylabel"
    :plot_type="plot_type"
    :title="title"
  />
  <div class="buttons">
    <button @click.stop="span = 30" :class="{'active': span == 30}">Month</button>
    <button @click.stop="span = 365" :class="{'active': span == 365}">Year</button>
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
  title?: string,
}>(), {
  plot_type: 'scatter',
});

const span = ref(30);
const slice_index = computed(() => datesBackIndex(span.value));
</script>

<style scoped>
.buttons {
  display: flex;
  justify-content: center;
  margin: 1rem;
}

button {
  font-size: .8rem;
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
</style>
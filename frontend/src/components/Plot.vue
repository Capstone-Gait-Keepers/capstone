<template>
  <div :id="_id" />
</template>

<script setup>
import Plotly from 'plotly.js-dist';
import { defineProps, onMounted } from 'vue';

// TODO: Add bounds

const { x, y, xlabel, ylabel, plot_type, title } = defineProps({
  x: Array,
  y: Array,
  xlabel: String,
  ylabel: String,
  plot_type: { plot_type: String, default: 'scatter' },
  title: { type: String | null, default: null },
});


function uuidv4() {
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}
const _id = uuidv4();

onMounted(() => {
  const data = [{x, y, type: plot_type}];
  const layout = {
    title,
    margin: {l: 32, r: 0, t: 0, b: 32},
    // paper_bgcolor: "#fff", // Transparent background
    // plot_bgcolor: "#fff", // Transparent plot area
    xaxis: {
      title: {
        text: xlabel,
        font: {
          size: 12,
          color: '#7f7f7f'
        }
      }
    },
    yaxis: {
      title: {
        text: ylabel,
        font: {
          size: 12,
          color: '#7f7f7f'
        }
      }
    }
  };
  const config = {displayModeBar: false};
  Plotly.newPlot(_id, data, layout, config);
});
</script>

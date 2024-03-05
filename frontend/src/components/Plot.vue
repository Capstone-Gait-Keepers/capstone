<template>
  <div id="plot" />
</template>

<script setup>
import Plotly from 'plotly.js-dist';
import { defineProps, onMounted } from 'vue';


const { x, y, xlabel, ylabel } = defineProps({
  x: Array,
  y: Array,
  xlabel: String,
  ylabel: String,
});


onMounted(() => {
  const data = [{x, y, type: 'scatter'}];
  const layout = {
    // title: 'A Fancy Plot',
    margin: {l: 32, r: 0, t: 0, b: 32}, // Remove margin
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
  Plotly.newPlot('plot', data, layout, config);
});
</script>

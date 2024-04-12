<template>
  <div :id="_id" />
</template>

<script setup>
import Plotly from 'plotly.js-dist';
import { onMounted, watch } from 'vue';

// TODO: Add bounds

const props = defineProps({
  x: Array,
  y: Array,
  xlabel: String,
  ylabel: String,
  minrange: { type: Number, default: 0.3 },
  hline: { type: Number, default: null },
  plot_type: { plot_type: String, default: 'scatter' },
  title: { type: String | null, default: null },
});


function uuidv4() {
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}
const _id = uuidv4();

function getLayout(y) {
  const maxY = Math.max(...y);
  const rangeMax = Math.max(props.minrange, props.hline * 1.1 + 0.1, maxY * 1.1 + 0.1);
  const minY = Math.min(...y);
  const rangeMin = Math.min(0, minY * 0.9 - 0.1);

  let layout = {
    title: props.title,
    margin: {r: 0, t: 0},
    xaxis: {
      title: {
        text: props.xlabel,
        font: {
          size: 14,
          color: '#000'
        }
      }
    },
    yaxis: {
      range: [rangeMin, rangeMax],
      title: {
        text: props.ylabel,
        font: {
          size: 14,
          color: '#000'
        }
      }
    },
  };
  if (props.hline !== null) {
    layout.shapes = [{
      type: 'line',
      xref: 'paper',
      x0: 0,
      y0: props.hline,
      x1: 1,
      y1: props.hline,
      line: {
        width: 2,
        dash:'dash',
      },
      name: 'Typical',
    }];
  }
  return layout;
}

const config = {displayModeBar: false, responsive: true};

function getTrace(x, y) {
  return {
    x, y,
    type: props.plot_type,
    mode: 'lines+markers',
    line: { shape: 'spline' },
    name: props.ylabel,
  };
}

onMounted(() => {
  Plotly.newPlot(_id, [getTrace(props.x, props.y)], getLayout(props.y), config);
});

watch([() => props.x, () => props.y], (newV, oldV) => {
  const [x, y] = newV;
  Plotly.newPlot(_id, [getTrace(x, y)], getLayout(y), config);
});
</script>

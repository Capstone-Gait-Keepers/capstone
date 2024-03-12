<template>
  <div class="alert-box" v-if="metric_type">
    <p class="exclamation">!</p>
    <div>
      <strong>Changes detected!</strong>
      <p>Negative changes were detected in your {{ props.metric_type }} over the last month. Consider setting an appointment up with your doctor to check in on your balance.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';

const props = defineProps({
  metric_type: {
    type: String,
    default: '',
  }
});
</script>

<style scoped>
.alert-box {
  padding: 0.5rem 1rem;
  color: black;
  border: 2px solid var(--yellow);
  border-radius: 8px;
  margin: 1rem 0;
  gap: 1rem;
  display: flex;
}

.exclamation {
  font-size: 3rem;
  color: var(--yellow);
  line-height: 1;
}
</style>
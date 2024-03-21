<template>
  <div class="alert-box" v-if="open && metric_type">
    <p class="exclamation">!</p>
    <div>
      <strong>Changes detected!</strong>
      <p>Negative changes were detected in your {{ props.metric_type }} over the last month. Consider setting an appointment up with your doctor to check in on your {{ props.metric_type }}.</p>
    </div>
    <p @click="open = false" class="close-button">X</p>
  </div>
</template>

<script setup lang="ts">
const props = defineProps({
  metric_type: {
    type: String,
    default: '',
  }
});

// v-model open close state
const open = defineModel<boolean>();

</script>

<style scoped>
.alert-box {
  padding: 0.5rem 1rem;
  color: black;
  background-color: white;
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

.close-button {
  font-size: 1.2rem;
  color: var(--yellow);
  cursor: pointer;
  margin-left: auto;
}
</style>
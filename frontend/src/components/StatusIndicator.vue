<template>
  <div class="status-indicator">
    <b>This {{ timespan }}</b>
    <img src="@/assets/icon-thumbs-up.svg" />
    <p>2 of 4 metrics showed positive changes compared with last month!</p>
  </div>
</template>

<script setup lang="ts">
import type { Metrics } from '@/types';

const { timespan, metrics } = defineProps<{timespan: "Month" | "Year", metrics: Metrics}>();



</script>

<style scoped>

</style>
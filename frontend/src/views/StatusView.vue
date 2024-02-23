<template>
  <BasePage>
    <h2>Sensor Status</h2>
    <table v-if="sensors !== null && sensors.length">
      <tr>
        <th>Sensor ID</th>
        <th>User ID</th>
        <th>Model</th>
        <th>Floor</th>
        <th>Last Timestamp</th>
        <th># of Recordings</th>
      </tr>
      <tr v-for="sensor in sensors">
        <td>{{ sensor.id }}</td>
        <td>{{ sensor.userid }}</td>
        <td>{{ sensor.model }}</td>
        <td>{{ sensor.floor }}</td>
        <td>{{ sensor.last_timestamp }}</td>
        <td>{{ sensor.num_recordings }}</td>
      </tr>
    </table>
    <p v-else-if="sensors !== null">No sensors loaded. Check console</p>
    <p v-else>Loading...</p>
  </BasePage>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import BasePage from '@/components/BasePage.vue'
import type { SensorConfig } from '@/types';
import { queryBackend } from '@/backend_interface';
const sensors = ref<SensorConfig[] | null>(null);

onMounted(async () => {
  sensors.value = await queryBackend('/api/sensor_status') || [];
});
</script>

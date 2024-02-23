<template>
  <main>
    <h2>Sensor Status</h2>
    <table>
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
  </main>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import type { SensorConfig } from '@/types';
import { queryBackend } from '@/backend_interface';
const sensors = ref<SensorConfig[]>([]);

onMounted(async () => {
  sensors.value = await queryBackend('/api/sensor_status');
});
</script>

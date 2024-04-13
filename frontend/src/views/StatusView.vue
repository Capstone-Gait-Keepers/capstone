<template>
  <main>
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
      <tr v-for="sensor in sensors" @click="selectSensor">
        <td><a>{{ sensor.id }}</a></td>
        <td>{{ sensor.userid }}</td>
        <td>{{ sensor.model }}</td>
        <td>{{ sensor.floor }}</td>
        <td>{{ sensor.last_timestamp }}</td>
        <td>{{ sensor.num_recordings }}</td>
      </tr>
    </table>
    <p v-else-if="sensors !== null">No sensors loaded. Check console</p>
    <p v-else>Loading...</p>

    <div v-if="recordingIds !== null && recordingIds.length">
      <h2>Recordings</h2>
      <div v-for="recordingId in recordingIds.reverse()">
        <a target="_blank" :href="getPlotUrl(recordingId)">{{ recordingId }}</a>
      </div>
    </div>
    <p v-else-if="recordingIds !== null">No available recordings</p>
  </main>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import type { SensorConfig } from '@/types';
import { queryBackend } from '@/backend_interface';
const sensors = ref<SensorConfig[] | null>(null);
const recordingIds = ref<Number[] | null>(null);

onMounted(async () => {
  sensors.value = await queryBackend('/api/sensor_status') || [];
});

async function selectSensor(event: MouseEvent) {
  if (event.currentTarget instanceof HTMLTableRowElement) {
    const sensor = sensors.value?.[event.currentTarget.rowIndex - 1];
    console.debug('Selected sensor:', sensor);
    if (sensor && sensor.id) {
      recordingIds.value = await queryRecordings(sensor.id);
    }
  }
}

async function queryRecordings(sensor_id: Number): Promise<Array<Number> | null> {
  const response = await queryBackend<Array<Number>>(`/api/list_recordings/${sensor_id}`);
  return response || null;
}

function getPlotUrl(rec_id: Number): string {
  return import.meta.env.VITE_BACKEND_URL + `/recording/${rec_id}`;
}
</script>

<style scoped>
main {
  display: flex;
  flex-direction: column;
  padding: 2rem;
  width: 60%;
}

a {
  cursor: pointer;
  text-decoration: underline;
  color: blue;
}
</style>

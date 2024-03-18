<script setup lang="ts">
import { RouterLink } from 'vue-router';

import BasePage from '@/components/BasePage.vue';
import ChangeAlert from '@/components/ChangeAlert.vue';
import StatusIndicator from '@/components/StatusIndicator.vue';
import { store, Section, validSection } from '@/store';
const { user } = store;
</script>

<template>
  <BasePage>
    <h1>Good Afternoon, {{ user?.name }}.</h1>
    <ChangeAlert metric_type="balance" />
    <h2>Welcome to your personal insights</h2>
    
    <div class="row" v-if="validSection(Section.Balance)">
      <h3>Balance Indicators</h3>
      <div class="row-grid">
        <div class="row-info">
          <p>WalkWise measures two metrics associated with balance.</p>
          <button><RouterLink to="/breakdown#Balance">View Breakdown</RouterLink></button>
        </div>
        <StatusIndicator :section="Section.Balance" timespan="Month" />
        <StatusIndicator :section="Section.Balance" timespan="Year" />
      </div>
    </div>

    <div class="row" v-if="validSection(Section.Neurodegenerative)">
      <h3>Neurodegenerative Indicators</h3>
      <div class="row-grid">
        <div class="row-info">
          <p>WalkWise measures three metrics associated with neurodegenerative diseases.</p>
          <button><RouterLink to="/breakdown#Neurodegenerative">View Breakdown</RouterLink></button>
        </div>
        <StatusIndicator :section="Section.Neurodegenerative" timespan="Month" />
        <StatusIndicator :section="Section.Neurodegenerative" timespan="Year" />
      </div>
    </div>

    <div class="row" v-if="validSection(Section.Dementia)">
      <h3>Dementia Indicators</h3>
      <div class="row-grid">
        <div class="row-info">
          <p>WalkWise measures four metrics associated with Dementia.</p>
          <button><RouterLink to="/breakdown#Dementia">View Breakdown</RouterLink></button>
        </div>
        <StatusIndicator :section="Section.Dementia" timespan="Month" />
        <StatusIndicator :section="Section.Dementia" timespan="Year" />
      </div>
    </div>

    <div class="row">
      <h3>Measurements Collected</h3>
      <div class="row-grid">
        <p>WalkWise needs to collect a certain amount of data for it to be reliable.</p>
        <div class="status-indicator">
          <b>This Month</b>
          <h3 class="measurement-stat">16</h3>
          <p>Measurements collected</p>
        </div>
        <div class="status-indicator">
          <b>This Year</b>
          <h3 class="measurement-stat">320</h3>
          <p>Measurements collected</p>
        </div>
      </div>
    </div>
    <div class="measurement-info">
      <h2>Last Measurement: February 21, 2024</h2>
      <p><a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank">Learn more</a> about how to set up your sensor</p>
    </div>
    <strong>Disclaimer:</strong>
    <i>
      This app is designed to collect gait metrics forinformational and self-tracking purposes only.
      The information provided by this app is not intended to replace professional medical advice,
      diagnosis, or treatment. Users are advised to consult with a qualified healthcare professional
      before making any decisions based on the data provided by this app, particularly if they have
      any concerns about their gait or overall health. This app does not diagnose medical conditions
      or provide medical treatment. Users should always seek the advice of their physician or other
      qualified healthcare provider with any questions they may have regarding a medical condition.
    </i>
  </BasePage>
</template>

<style scoped>
.row {
  padding: 3rem 2rem;
  background-color: white;
  border: 2px solid var(--color-main);
  border-radius: 1rem;
  box-shadow: 0 4px 4px 0 rgba(0, 0, 0, 0.25);
  margin: 1rem 0;
}

.row-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
}

.row-info {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

h3 {
  font-weight: bold;
}

button {
  margin: 1rem 0;
  width: fit-content;
}

.measurement-info {
  padding: 2rem 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  border-bottom: 4px solid var(--color-main);
  margin-bottom: 1rem;
}

.measurement-stat {
  font-size: 2rem;
  font-weight: bold;
  color: var(--color-main);
}
</style>

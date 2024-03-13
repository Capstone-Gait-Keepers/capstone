<script setup lang="ts">
import BasePage from '@/components/BasePage.vue';
import ChangeAlert from '@/components/ChangeAlert.vue';
import { RouterLink } from 'vue-router';
</script>

<template>
  <BasePage>
    <h1>Good Afternoon Marjorie</h1>
    <ChangeAlert metric_type="balance" />
    <h2>Welcome to your personal insights</h2>
    
    <div class="metrics-row">
      <h3>Balance Metrics</h3>
      <div class="metrics-row-grid">
        <div>
          <p>WalkWise measures two metrics associated with balance.</p>
          <button><RouterLink to="/breakdown#dementia">View Breakdown</RouterLink></button>
        </div>
        <div class="status-indicator">
          <p>This Month</p>
          <img src="@/assets/icon-thumbs-up.svg" />
          <p>2 of 4 metrics showed positive changes compared with last month!</p>
        </div>
        <div class="status-indicator">
          <p>This Year</p>
          <img src="@/assets/icon-thumbs-up.svg" />
          <p>2 of 4 metrics showed positive changes compared with last year!</p>
        </div>
      </div>
    </div>

    <div class="metrics-row">
      <h3>Neurodegenerative Metrics</h3>
      <div class="metrics-row-grid">
        <div>
          <p>WalkWise measures two metrics associated with neurodegenerative diseases.</p>
          <button><RouterLink to="/breakdown#neurodegenerative">View Breakdown</RouterLink></button>
        </div>
        <div class="status-indicator">
          <p>This Month</p>
          <div class="flat-bar"></div>
          <p>No changes in neurodegenerative related metrics this month.</p>
        </div>
        <div class="status-indicator">
          <p>This Year</p>
          <img src="@/assets/icon-warning.svg" />
          <p>There were negative changes in 2 of the 3 metrics related to neurodegenerative disease.</p>
        </div>
      </div>
    </div>

    <div class="metrics-row">
      <h3>Dementia Metrics</h3>
      <div class="metrics-row-grid">
        <div>
          <p>WalkWise measures 4 metrics associated with Dementia.</p>
          <button><RouterLink to="/breakdown#dementia">View Breakdown</RouterLink></button>
        </div>
        <div class="status-indicator">
          <p>This Month</p>
          <div class="flat-bar"></div>
          <p>No changes in Dementia related metrics this month.</p>
        </div>
        <div class="status-indicator">
          <p>This Year</p>
          <div class="flat-bar"></div>
          <p>No changes in Dementia related metrics compared with last year.</p>
        </div>
      </div>
    </div>
    <div class="data-collection-info">
      <h2>Last data point collected: Blah</h2>
      <p>Learn more about how to set up your sensor</p>
    </div>
    <strong>Disclaimer:</strong>
    <p>
      This app is designed to collect gait metrics forinformational and self-tracking purposes only.
      The information provided by this app is not intended to replace professional medical advice,
      diagnosis, or treatment. Users are advised to consult with a qualified healthcare professional
      before making any decisions based on the data provided by this app, particularly if they have
      any concerns about their gait or overall health. This app does not diagnose medical conditions
      or provide medical treatment. Users should always seek the advice of their physician or other
      qualified healthcare provider with any questions they may have regarding a medical condition.
    </p>
  </BasePage>
</template>

<style scoped>
.metrics-row {
  padding: 3rem 2rem;
  background-color: #FBFBFB;
  margin: 1rem 0;
}

.metrics-row-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
}

h3 {
  font-weight: bold;
}

button {
  margin: 1rem 0;
  background-color: var(--color-main);
}

.flat-bar {
  border-bottom: 4px solid var(--color-main);
  width: 80px;
  height: 0;
  margin: 1rem 0;
}

.status-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 0 2rem;
}

.status-indicator img {
  width: 4em;
  height: 4em;
}

.data-collection-info {
  padding: 2rem 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}
</style>

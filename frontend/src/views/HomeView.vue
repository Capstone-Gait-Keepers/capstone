<script setup lang="ts">
import BasePage from '@/components/BasePage.vue'
import ChangeAlert from '@/components/ChangeAlert.vue';
</script>

<template>
  <BasePage>
    <h1>Good Afternoon Marjorie</h1>
    <ChangeAlert />
    <div class="insights-container">
      <h2>Welcome to your personal insights</h2>

      <h3>Balance Metrics</h3>
      <div class="metrics-container">
        <div>
          <p>WalkWise measures two metrics associated with balance.</p>
          <ul>
            <li>Stride Time Coefficient of Variation</li>
            <li>Stride Time Gait Asymmetry</li>
          </ul>
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

      <h3>Neurodegenerative Metrics</h3>
      <div class="metrics-container">
        <div>
          <p>WalkWise measures two metrics associated with Neurodegenerative diseases.</p>
          <ul>
            <li>Stride Time Phase Synchronization</li>
            <li>Stride Time Conditional Entropy</li>
            <li>Stride Time Gait Asymmetry</li>
          </ul>
        </div>
        <div class="status-indicator">
          <p>This Month</p>
          <div class="flat-bar"></div>
          <p>No changes in neurodegenerative related metrics this month.</p>
        </div>
        <div class="status-indicator">
          <p>This Year</p>
          <img src="@/assets/icon-warning.svg" />
          <p>There were negative changes in 2 of the 3 metrics related to neurodegenerative disease. View the <a href="#">Breakdown</a> to see more.</p>
        </div>
      </div>

      <h3>Dementia Metrics</h3>
      <div class="metrics-container">
        <div>
          <p>WalkWise measures 4 metrics associated with Dementia.</p>
          <ul>
            <li>Stride Time</li>
            <li>Cadence</li>
            <li>Stride Time Coefficient of Variation</li>
            <li>Swing Time Gait Asymmetry</li>
          </ul>
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
  </BasePage>
</template>

<style scoped>
.insights-container {
  padding: 1rem 6rem;
}

.metrics-container {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin-bottom: 40px;
}

h3 {
  font-weight: bold;
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
  gap: 10px;
}

.status-indicator img {
  width: 50px;
  height: 50px;
}
</style>

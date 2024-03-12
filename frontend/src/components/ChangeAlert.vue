<template>
  <div class="alert-box">
    <p class="exclamation">!</p>
    <div>
      <strong>Changes detected!</strong>
      <p>Negative changes were detected in your Balance over the last month. Consider setting an appointment up with your doctor to check in on your balance.</p>
    </div>
  </div>
</template>

<style scoped>
.alert-box {
  padding: 0.5rem 1rem;
  color: black;
  border: 2px solid var(--yellow);
  border-radius: 8px;
  margin: 1rem;
  gap: 1rem;
  display: flex;
}

.exclamation {
  font-size: 3rem;
  color: var(--yellow);
  line-height: 1;
}

strong {
  font-weight: bold;
}
</style>
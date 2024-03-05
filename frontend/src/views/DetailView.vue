<script setup lang="ts">
import BasePage from '@/components/BasePage.vue'
import Plot from '@/components/Plot.vue'
</script>

<template>
  <BasePage>
    <h1>Learn More</h1>
    <p>Here's a fancy plot:</p>
    <Plot :x="[2, 3, 4]" :y="[2, 3, 2]" xlabel="X" ylabel="Y" />
  </BasePage>
</template>

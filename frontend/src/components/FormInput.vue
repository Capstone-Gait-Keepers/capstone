<template>
  <div class="form-input">
    <label for="input">{{ label }}</label>
    <div class="input-button">
      <input :type="type" :disabled="!editable" v-model="model" />
      <button @click="editable = !editable">{{ editable ? 'Save' : 'Edit' }}</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const { label, type } = defineProps<{
  label: string;
  type: string;
}>();
const model = defineModel();
const editable = ref(false);
</script>

<style scoped>
.form-input {
  display: flex;
  flex-direction: column;
  margin-bottom: 1rem;
}

.input-button {
  display: flex;
}

input {
  border: 1px solid var(--color-main);
  border-radius: 8px;
  padding: 0.6rem 1rem;
  width: 30rem;
  max-width: 100%;
  margin-right: 2rem;
}
</style>
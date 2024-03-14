<template>
  <div class="form-input">
    <label v-if="label" :for="label">{{ label }}</label>
    <div class="input-button">
      <input :type="type" :disabled="!editable" v-model="model" :id="label" />
      <button @click="click">{{ editable ? 'Save' : 'Edit' }}</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const { label, type, startEditing } = defineProps<{
  label?: string;
  type: string;
  startEditing?: boolean;
}>();
const model = defineModel<string | number>();
const editable = ref(startEditing || false);
const emit = defineEmits<{save: [input: string | number]}>();

function click() {
  if (editable.value && model.value) {
    emit('save', model.value);
  }
  editable.value = !editable.value;
}
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
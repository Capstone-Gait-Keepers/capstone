<template>
  <div >
    <button @click="toggleAccordion" class="header">
      {{ header }}
    </button>
    <transition
      @enter="start"
      @after-enter="end"
      @before-leave="start"
      @after-leave="end"
    >
      <div
        v-if="isOpen"
        class="accordion-content"
      >
        <slot></slot>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onMounted, defineEmits } from 'vue';

const props = defineProps<{isOpen: boolean, header: string}>();
const emit = defineEmits(['update:isOpen']);

function toggleAccordion() {
  emit('update:isOpen', !props.isOpen);
}
function start(el: HTMLElement) {
  el.style.height = el.scrollHeight + "px";
}
function end(el: HTMLElement) {
  el.style.height = "";
}
</script>

<style>
.header {
  z-index: 100;
}

.accordion-content {
  z-index: -1;
}

.v-enter-active,
.v-leave-active {
  will-change: height, opacity;
  transition: all 0.5s ease;
  overflow: hidden;
}

.v-enter-from,
.v-leave-to {
  z-index: 1;
  opacity: 0;
  transform: translateY(-100%);
  height: 0 !important;
}
</style>

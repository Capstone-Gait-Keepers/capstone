<template>
  <div @click="isOpen = !isOpen" class="main">
    <div class="header">
      {{ header }}
      <svg :class="{flipped: isOpen}" width="33" height="22" viewBox="0 0 33 22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 3L16.5624 17L29.5563 3" stroke="#3481B9" stroke-width="6" stroke-linecap="round"/>
      </svg>
    </div>
    <transition
      @enter="start"
      @after-enter="end"
      @before-leave="start"
      @after-leave="end"
    >
      <div v-if="isOpen" class="accordion-content">
        <slot></slot>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const { header } = defineProps<{header?: string}>();
const isOpen = ref(false);

function start(el: any) {
  el.style.height = el.scrollHeight + "px";
}
function end(el: any) {
  el.style.height = "";
}
</script>

<style scoped>
.main {
  padding: 1rem;
  display: flex;
  flex-direction: column;

  background-color: white;
  border: 2px solid var(--color-main);
  border-radius: 1rem;
  box-shadow: 0 4px 4px 0 rgba(0, 0, 0, 0.25);
  cursor: pointer;

}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
  font-size: 1.2rem;
  color: var(--color-main);
}

.header svg {
  width: 1em;
  transition: transform 0.5s;
}

.accordion-content {
  overflow: hidden;
}

.flipped {
  transform: scaleY(-1);
}

.flip-enter-active .header svg {
  transform: rotate(180deg);
}

.v-enter-active,
.v-leave-active {
  transition: opacity 0.5s ease, height 0.5s ease;
  overflow: hidden;
}

.v-enter-from,
.v-leave-to {
  opacity: 0;
  height: 0 !important;
}
</style>

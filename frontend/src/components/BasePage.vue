<script setup lang="ts">
import { RouterLink } from 'vue-router'
import Logo from '@/components/Logo.vue'
</script>

<template>
  <div>
    <header>
      <Logo />
      <nav>
        <RouterLink to="/"><img src="@/assets/icon-overview.svg" alt="overview-icon"/>Overview</RouterLink>
        <RouterLink to="/breakdown"><img src="@/assets/icon-breakdown.svg" alt="breakdown-icon"/>Breakdown</RouterLink>
        <RouterLink to="/settings"><img src="@/assets/icon-settings.svg" alt="settings-icon"/>Settings</RouterLink>
        <!-- <RouterLink to="/status">Status</RouterLink> -->
        <button><RouterLink to="/login">Sign out</RouterLink></button>
      </nav>
    </header>
    <main>
      <slot/>
    </main>
  </div>
</template>

<style scoped>
header {
  position: sticky;
  top: 0;
  border-bottom: 2px solid var(--color-main);
  z-index: 1000;
  display: flex;
  justify-content: space-between;
  background-color: white;
}

main {
  margin: 1rem 3rem;
}

nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav button {
  margin-left: 1rem;
  background-color: var(--red);
}

nav a {
  display: flex;
  align-items: center;
  padding: 0 1rem;
  gap: .5rem;
}

nav a img {
  height: 1.2em;
  width: 1.2em;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding: 0 1rem;
  }

  nav {
    text-align: left;
    margin: .2rem;
    padding: .25rem 0;
  }
}
</style>

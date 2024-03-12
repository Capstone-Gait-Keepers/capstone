<script setup lang="ts">
import { RouterLink } from 'vue-router'

import overviewIcon from "@/assets/icon-overview.svg";
import breakdownIcon from "@/assets/icon-breakdown.svg";
import settingsIcon from "@/assets/icon-settings.svg";
</script>

<template>
  <div>
    <header>
      <div class="logo">
        <img alt="WalkWise Logo" class="logo" src="@/assets/logo.svg" width="125" height="125" />
        <p>WalkWise</p>
      </div>
      <nav>
        <RouterLink to="/"><img :src="overviewIcon" alt="overview-icon"/>Overview</RouterLink>
        <RouterLink to="/breakdown"><img :src="breakdownIcon" alt="breakdown-icon"/>Breakdown</RouterLink>
        <RouterLink to="/settings"><img :src="settingsIcon" alt="settings-icon"/>Settings</RouterLink>
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
  font-size: 1.2em;
}

.logo {
  display: flex;
  align-items: center;
  height: 2rem;
  width: auto;
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
  background-color: var(--red);
  color: white;
  border: none;
  cursor: pointer;
  padding: .5rem 0;
  border-radius: 6px;
  box-shadow: 1px 2px 5px rgba(0, 0, 0, 0.5);
  margin-left: 1rem;
}

nav button a {
  color: white;
  text-decoration: none;
  background-color: transparent;
}

nav a {
  display: flex;
  align-items: center;
  padding: 0 1rem;
  gap: .5rem;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding: 0 1rem;
  }

  .logo {
    margin-right: .5rem;
  }

  nav {
    text-align: left;
    margin: .2rem;
    padding: .25rem 0;
  }
}
</style>

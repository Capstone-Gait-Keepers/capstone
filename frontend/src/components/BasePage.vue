<script setup lang="ts">
import { RouterLink } from 'vue-router'
</script>

<template>
  <div class="wrapper">
    <header>
      <div>
        <img alt="WalkWise Logo" class="logo" src="@/assets/logo.svg" width="125" height="125" />
        <p>WalkWise</p>
      </div>
      <nav>
        <RouterLink to="/">Insights</RouterLink>
        <RouterLink to="/detail">Learn More</RouterLink>
        <RouterLink to="/settings">Settings</RouterLink>
        <RouterLink to="/status">Status</RouterLink>
        <RouterLink to="/login">Sign out</RouterLink>
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
  border-bottom: 1px solid #ccc;
  z-index: 1000;
  display: flex;
  justify-content: space-between;
}

.logo {
  display: block;
  height: 3rem;
  width: auto;
  margin: 0 auto .5rem;
}

nav {
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem 0;
    margin-top: 1rem;
  }
}
</style>
